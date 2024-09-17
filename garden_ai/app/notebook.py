import logging
import shutil
import webbrowser
from pathlib import Path
from typing import Optional
import json
import time
import os

import docker  # type: ignore
import typer
from tempfile import TemporaryDirectory

from garden_ai import GardenClient, GardenConstants
from garden_ai.app.console import print_err
from garden_ai.containers import (
    build_image_with_dependencies,
    build_notebook_session_image,
    push_image_to_public_repo,
    start_container_with_notebook,
    get_docker_client,
    extract_metadata_from_image,
    DockerStartFailure,
    DockerBuildFailure,
    DockerPreBuildFailure,
)

from garden_ai.notebook_metadata import (
    add_notebook_metadata,
    set_notebook_metadata,
    get_notebook_metadata,
    read_requirements_data,
)

from garden_ai.utils.notebooks import (
    clear_cells,
    is_over_size_limit,
    generate_botanical_filename,
)

logger = logging.getLogger()

notebook_app = typer.Typer(name="notebook")

BASE_IMAGE_NAMES = ", ".join(
    ["'" + image_name + "'" for image_name in GardenConstants.PREMADE_IMAGES.keys()]
)


class DockerClientSession:
    def __init__(self, verbose=False) -> None:
        self.verbose = verbose

    def __enter__(self) -> docker.DockerClient:
        try:
            return get_docker_client()
        except DockerStartFailure as e:
            # We're most likely to see this error raised from get_docker_client.
            self.handle_docker_start_failure(e)
        except docker.errors.BuildError as e:
            self.handle_docker_build_failure(e)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is DockerStartFailure:
            # If the user's Docker daemon shuts down partway through the session
            # and another docker command is issued, we'll catch that here.
            self.handle_docker_start_failure(exc_val)
        # Use isinstance to catch subclasses of docker.errors.BuildError
        elif isinstance(exc_val, docker.errors.BuildError):
            self.handle_docker_build_failure(exc_val)

    def handle_docker_build_failure(self, e: docker.errors.BuildError):
        # If the user is in verbose mode, the build log has already been printed.
        if not self.verbose:
            for line in e.build_log:
                typer.echo(line)

        print_err(f"Fatal Docker build error: {e}\n" "Above is the full build log.\n")

        if isinstance(e, DockerPreBuildFailure):
            print_err(
                "Garden could not set up your base Docker image. "
                "If you supplied a requirements file, check that it's formatted correctly.\n"
            )
        elif isinstance(e, DockerBuildFailure):
            last_line = e.build_log[-2] if len(e.build_log) > 1 else ""
            if "Traceback" in last_line:
                print_err(
                    "Garden could not build a Docker image from your notebook. "
                    "This is likely because of a bug in your notebook code.\n"
                    "This is where the error occurred: "
                )
                typer.echo(last_line)
            else:
                print_err(
                    "Garden could not build a Docker image from your notebook. "
                    "It looks like it is not an error in your notebook code.\n"
                )

        raise typer.Exit(1)

    def handle_docker_start_failure(self, e: DockerStartFailure):
        print_err("Garden can't access Docker on your computer.")
        if e.helpful_explanation:
            print_err(e.helpful_explanation)
            print_err(
                "If that doesn't work, use `garden-ai docker check` to troubleshoot."
            )
        else:
            print_err(
                "This doesn't look like one of the typical error cases. Printing error from Docker:"
            )
            typer.echo(e.original_exception)
        raise typer.Exit(1)


@notebook_app.callback(no_args_is_help=True)
def notebook():
    """sub-commands for editing and publishing from sandboxed notebooks."""
    pass


@notebook_app.command()
def list_premade_images():
    """List all Garden base docker images"""
    print(f"Garden premade images:\n{BASE_IMAGE_NAMES}")


@notebook_app.command(no_args_is_help=True)
def start(
    path: Optional[Path] = typer.Argument(
        default=None,
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
        help=("Path to a .ipynb notebook to open in a fresh, isolated container."),
    ),
    base_image_name: Optional[str] = typer.Option(
        None,
        "--base-image",
        help=(
            "A Garden base image to boot the notebook in. "
            "For example, to boot your notebook with the default Garden python 3.8 image, use --base-image 3.8-base. "
            "To see all the available Garden base images, use 'garden-ai notebook list-premade-images'"
        ),
    ),
    requirements_path: Optional[Path] = typer.Option(
        None,
        "--requirements",
        help=(
            "Path to a requirements.txt containing "
            "additional dependencies to install in the base image."
        ),
    ),
    global_notebook_doi: Optional[str] = typer.Option(
        None,
        "--doi",
        help=(
            "DOI of a Garden you want to add all entrypoints in this notebook to. "
            "To override the global notebook DOI for a specific entrypoint, "
            "provide the entrypoint decorator with the optional garden_doi argument."
        ),
    ),
    custom_image_uri: Optional[str] = typer.Option(
        None,
        "--custom-image",
        help=(
            "Power users only! Provide a uri of a publicly available docker image to boot the notebook in."
        ),
        hidden=True,
    ),
    tutorial: Optional[bool] = typer.Option(
        False,
        "--tutorial",
        help=(
            "First time using Garden? Open this notebook that walks you through publishing your first model."
        ),
        hidden=True,
    ),
):
    """Open a notebook file in a sandboxed environment. Optionally, specify a different base docker image.

    Changes to the notebook file will persist after the container shuts down.
    Quit the process with Ctrl-C or by shutting down jupyter from the browser.
    If a different base image is chosen, that image will be reused as the default for this notebook in the future.
    """

    # First figure out the name of the notebook and whether we need to create it
    need_to_create_notebook = False

    if path is None:
        need_to_create_notebook = True
        new_notebook_name = generate_botanical_filename()
        notebook_path = Path.cwd() / new_notebook_name
    else:
        notebook_path = path.resolve()
        if notebook_path.suffix != ".ipynb":
            typer.echo("File must be a jupyter notebook (.ipynb)")
            raise typer.Exit(1)

        if not notebook_path.exists():
            need_to_create_notebook = True

    # Figure out what base image uri we should start the notebook in
    base_image_uri = _get_base_image_uri(
        base_image_name,
        custom_image_uri,
        None if need_to_create_notebook else notebook_path,
    )

    # Build prompt
    if need_to_create_notebook:
        message = f"This will create a new notebook {notebook_path.name} and open it in Docker image {base_image_uri}.\n"
    else:
        message = f"This will open existing notebook {notebook_path.name} in Docker image {base_image_uri}.\n"

    if requirements_path:
        message += f"Additional dependencies specified in {requirements_path.name} will also be installed in {base_image_uri}.\n"
        message += "Any dependencies previously associated with this notebook will be overwritten by the new requirements.\n"

    # Now we have all we need to prompt the user to proceed
    typer.confirm(message + "Do you want to proceed?", abort=True)

    if need_to_create_notebook:
        if tutorial:
            template_file_name = "tutorial.ipynb"
        elif base_image_name:
            template_file_name = GardenConstants.IMAGES_TO_FLAVOR.get(
                base_image_name, "empty.ipynb"
            )
        else:
            template_file_name = "empty.ipynb"

        top_level_dir = Path(__file__).parent.parent
        source_path = top_level_dir / "notebook_templates" / template_file_name
        shutil.copy(source_path, notebook_path)

    # Adds metadata widget cell and garden_metadata dict if either is missing
    add_notebook_metadata(notebook_path)

    notebook_metadata = get_notebook_metadata(notebook_path)

    # Validate and read requirements file.
    if requirements_path:
        requirements_path = requirements_path.resolve()
        _validate_requirements_path(requirements_path)
        requirements_data = read_requirements_data(requirements_path)
    else:
        # If no requirements file given, look for requirements data in notebook metadata
        requirements_data = notebook_metadata.notebook_requirements

    # Check for base image name from notebook if user did not provide one.
    if base_image_name is None:
        # If a user is using a custom image uri, base_image_name might be None.
        base_image_name = notebook_metadata.notebook_image_name

    # Check for global notebook doi from notebook if user did not provide one.
    if global_notebook_doi is None:
        global_notebook_doi = notebook_metadata.global_notebook_doi

    # Update garden metadata in notebook
    set_notebook_metadata(
        notebook_path,
        global_notebook_doi,
        base_image_name,
        base_image_uri,
        requirements_data,
    )

    print(
        f"Starting notebook inside base image with full name {base_image_uri}. "
        f"If you start this notebook again from the same folder, it will use this image by default."
    )

    with DockerClientSession(verbose=True) as docker_client:
        # pre-bake local image with garden-ai and additional user requirements
        local_base_image_id = build_image_with_dependencies(
            docker_client, base_image_uri, requirements_data
        )
        # start container and listen for Ctrl-C
        container = start_container_with_notebook(
            docker_client,
            notebook_path,
            local_base_image_id,
            requirements_path,
            pull=False,
        )
        _register_container_sigint_handler(container)

    container.reload()
    port = container.attrs["NetworkSettings"]["Ports"]["8888/tcp"][0]["HostPort"]
    typer.echo(
        f"Notebook started! Opening http://127.0.0.1:{port}/notebooks/{notebook_path.name} "
        "in your default browser (you may need to refresh the page)"
    )

    if not os.environ.get("GARDEN_DISABLE_BROWSER"):
        # Give the notebook server a few seconds to start up so that the user doesn't have to refresh manually
        time.sleep(3)
        webbrowser.open_new_tab(
            f"http://127.0.0.1:{port}/notebooks/{notebook_path.name}"
        )

    # stream logs from the container
    for line in container.logs(stream=True):
        print(line.decode("utf-8", errors="ignore"), end="")

    # block until the container finishes
    try:
        container.reload()
        container.wait()
    except KeyboardInterrupt:
        # handle windows Ctrl-C
        typer.echo("Stopping notebook ...")
        container.stop()
    except docker.errors.NotFound:
        # container already killed, no need to wait
        pass

    typer.echo("Notebook has stopped.")
    return


@notebook_app.command(no_args_is_help=True)
def debug(
    path: Path = typer.Argument(
        ...,
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
        exists=True,
        help=(
            "Path to a .ipynb notebook whose remote environment will be approximated for debugging."
        ),
    ),
    requirements_path: Optional[Path] = typer.Option(
        None,
        "--requirements",
        help=(
            "Path to a requirements.txt containing "
            "additional dependencies to install in the base image."
        ),
    ),
):
    """Open the debugging notebook in a pre-prepared container.

    Changes to the notebook file will NOT persist after the container shuts down.
    Quit the process with Ctrl-C or by shutting down jupyter from the browser.
    """

    with DockerClientSession(verbose=True) as docker_client:
        base_image_uri = (
            _get_base_image_uri(
                base_image_name=None, custom_image_uri=None, notebook_path=path
            )
            or "gardenai/base:python-3.10-base"
        )

        notebook_metadata = get_notebook_metadata(path)

        # Validate and read requirements file.
        if requirements_path:
            _validate_requirements_path(requirements_path)
            requirements_data = read_requirements_data(requirements_path)
        else:
            # If no requirements file given, look for requirements data in notebook metadata
            requirements_data = notebook_metadata.notebook_requirements

        base_image_name = notebook_metadata.notebook_image_name
        if base_image_name is None:
            base_image_name = "3.10-base"
        global_notebook_doi = notebook_metadata.global_notebook_doi

        with TemporaryDirectory() as temp_dir:
            # pre-bake local image with garden-ai and additional user requirements
            local_base_image_id = build_image_with_dependencies(
                docker_client, base_image_uri, requirements_data
            )

            # pass global notebook doi to image as env_var
            image = build_notebook_session_image(
                docker_client,
                path,
                local_base_image_id,
                env_vars={"GLOBAL_NOTEBOOK_DOI": global_notebook_doi},
            )

            if image is None:
                typer.echo("Failed to build image.")
                raise typer.Exit(1)
            image_name = str(image.id)  # str used to guarantee type-check

            top_level_dir = Path(__file__).parent.parent
            debug_path = top_level_dir / "notebook_templates" / "debug.ipynb"

            # Make tmp copy of debug notebook template to add original notebook's metadata too
            temp_debug_path = Path(temp_dir) / "debug.ipynb"
            shutil.copy(debug_path, temp_debug_path)
            set_notebook_metadata(
                temp_debug_path,
                global_notebook_doi,
                base_image_name,
                base_image_uri,
                requirements_data,
            )

            container = start_container_with_notebook(
                docker_client,
                temp_debug_path,
                image_name,
                requirements_path=None,
                mount=False,
                pull=False,
                custom_config=False,
            )
            _register_container_sigint_handler(container)

    container.reload()
    port = container.attrs["NetworkSettings"]["Ports"]["8888/tcp"][0]["HostPort"]
    typer.echo(
        f"Notebook started! Opening http://127.0.0.1:{port}/notebooks/{debug_path.name} "
        "in your default browser (you may need to refresh the page)"
    )
    webbrowser.open_new_tab(f"http://127.0.0.1:{port}/notebooks/{debug_path.name}")

    # stream logs from the container
    for line in container.logs(stream=True):
        print(line.decode("utf-8", errors="ignore"), end="")

    # block until the container finishes
    try:
        container.reload()
        container.wait()
    except KeyboardInterrupt:
        # handle windows Ctrl-C
        typer.echo("Stopping notebook ...")
        container.stop()
    except docker.errors.NotFound:
        # container already killed, no need to wait
        pass

    typer.echo("Notebook has stopped.")
    return


@notebook_app.command(no_args_is_help=True)
def publish(
    path: Path = typer.Argument(
        ...,
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
    ),
    requirements_path: Optional[Path] = typer.Option(
        None,
        "--requirements",
        help=(
            "Path to a requirements.txt containing "
            "additional dependencies to install in the base image."
        ),
    ),
    base_image_name: Optional[str] = typer.Option(
        None,
        "--base-image",
        help=(
            "A Garden base image to run your notebook inside of. This will be the foundation for the image that runs your entrypoints."
            "For example, to run on top of the default Garden python 3.8 image, use --base-image 3.8-base. "
            "To see all the available Garden base images, use 'garden-ai notebook list-premade-images'"
        ),
    ),
    global_notebook_doi: Optional[str] = typer.Option(
        None,
        "--doi",
        help=(
            "DOI of a Garden you want to publish all entrypoints in this notebook too. "
            "To override the global notebook DOI for a specific entrypoint, "
            "provide the entrypoint decorator with the optional garden_doi argument."
        ),
    ),
    custom_image_uri: Optional[str] = typer.Option(
        None,
        "--custom-image",
        help=(
            "Power users only! Provide a uri of a publicly available docker image to boot the notebook in."
        ),
        hidden=True,
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
    keep_outputs: bool = typer.Option(
        False,
        "--keep-outputs",
        help="By default, Garden will clear all cell outputs before publishing. "
        "If you would like to have your cell outputs visible on the UI, use this flag.",
    ),
):
    client = GardenClient()
    notebook_path = path.resolve()
    if notebook_path.suffix != ".ipynb":
        raise ValueError("File must be a jupyter notebook (.ipynb)")
    if not notebook_path.exists():
        raise ValueError(f"Could not find file at {notebook_path}")

    # Use tmpdir and make a copy of original notebook, since publish should not modify
    # the original notebook if provided with any new arguments.
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        tmp_notebook_path = temp_dir_path / notebook_path.name
        shutil.copy(notebook_path, tmp_notebook_path)

        base_image_uri = _get_base_image_uri(
            base_image_name, custom_image_uri, tmp_notebook_path
        )
        print(f"Using base image: {base_image_uri}")

        notebook_metadata = get_notebook_metadata(tmp_notebook_path)
        # Validate and read requirements file.
        if requirements_path:
            _validate_requirements_path(requirements_path)
            requirements_data = read_requirements_data(requirements_path)
        else:
            # If no requirements file given, look for requirements data in notebook metadata
            requirements_data = notebook_metadata.notebook_requirements

        # Check for base image name from notebook if user did not provide one.
        if base_image_name is None:
            # If a user is using a custom image uri, base_image_name might be None
            base_image_name = notebook_metadata.notebook_image_name

        # Check for global notebook doi from notebook if user did not provide one.
        if global_notebook_doi is None:
            global_notebook_doi = notebook_metadata.global_notebook_doi

        # Update the tmp notebooks metadata with any new publish args
        set_notebook_metadata(
            tmp_notebook_path,
            global_notebook_doi,
            base_image_name,
            base_image_uri,
            requirements_data,
        )

        # Pre-process the notebook and make sure it's not too big
        raw_notebook_contents = tmp_notebook_path.read_text()
        try:
            notebook_contents = json.loads(raw_notebook_contents)
        except json.JSONDecodeError:
            typer.echo("Could not parse notebook JSON.")
            raise typer.Exit(1)

        if not keep_outputs:
            notebook_contents = clear_cells(notebook_contents)

        if is_over_size_limit(notebook_contents):
            typer.echo("Garden can't publish notebooks bigger than 5MB.")
            raise typer.Exit(1)

        # Push the notebook to the Garden API
        notebook_url = client.upload_notebook(notebook_contents, tmp_notebook_path.name)

        with DockerClientSession(verbose=verbose) as docker_client:
            # Build the image
            local_base_image_id = build_image_with_dependencies(
                docker_client,
                base_image_uri,
                requirements_data,
                print_logs=verbose,
                pull=True,
            )

            image = build_notebook_session_image(
                docker_client,
                tmp_notebook_path,
                local_base_image_id,
                print_logs=verbose,
                env_vars={"GLOBAL_NOTEBOOK_DOI": global_notebook_doi},
            )

            if image is None:
                typer.echo("Failed to build image.")
                raise typer.Exit(1)
            typer.echo(f"Built image: {image}")

        # push image to ECR
        auth_config = client._get_auth_config_for_ecr_push()

        typer.echo(f"Pushing image to repository: {GardenConstants.GARDEN_ECR_REPO}")
        full_image_uri = push_image_to_public_repo(
            docker_client, image, auth_config, print_logs=verbose
        )
        typer.echo(f"Successfully pushed image to: {full_image_uri}")

        metadata = extract_metadata_from_image(docker_client, image)
        client._register_and_publish_from_user_image(
            base_image_uri, full_image_uri, notebook_url, metadata
        )


def _get_base_image_uri(
    base_image_name: Optional[str],
    custom_image_uri: Optional[str],
    notebook_path: Optional[Path],
) -> str:
    # First make sure that we have enough information to get a base image uri
    if base_image_name and custom_image_uri:
        typer.echo(
            "You specified both a base image and a custom image. Please specify only one."
        )
        raise typer.Exit(1)

    # Check for saved_image_name in notebooks metadata
    if notebook_path:
        saved_image_name = get_notebook_metadata(notebook_path).notebook_image_name
    else:
        saved_image_name = None

    if not any([base_image_name, custom_image_uri, saved_image_name]):
        typer.echo(
            f"Please specify a base image. The current Garden base images are: \n{BASE_IMAGE_NAMES}"
        )
        raise typer.Exit(1)

    # Now use precedence rules to get the base image uri
    # 1: --custom-image wins if specified
    if custom_image_uri:
        return custom_image_uri

    # 2: then go off of --base-image
    if base_image_name:
        if base_image_name in GardenConstants.PREMADE_IMAGES:
            return GardenConstants.PREMADE_IMAGES[base_image_name]
        else:
            typer.echo(
                f"The image you specified ({base_image_name}) is not one of the Garden base images. "
                f"The current Garden base images are: \n{BASE_IMAGE_NAMES}"
            )
            raise typer.Exit(1)

    # saved_image_name is definitely non-None at this point
    if saved_image_name in GardenConstants.PREMADE_IMAGES:
        saved_image_uri = GardenConstants.PREMADE_IMAGES[saved_image_name]
    else:
        typer.echo(
            f"The base image name ({saved_image_name}) saved in this notebook is not one of the Garden base images. "
            f"The current Garden base images are: \n{BASE_IMAGE_NAMES}"
        )
        raise typer.Exit(1)

    # 3: If the user didn't specify an image explicitly, use the the image name saved in the notebook.
    return saved_image_uri


def _register_container_sigint_handler(container: docker.models.containers.Container):
    """helper: ensure SIGINT/ Ctrl-C to our CLI stops a given container"""
    import signal

    def handler(signal, frame):
        typer.echo("Stopping notebook...")
        container.stop()
        return

    signal.signal(signal.SIGINT, handler)
    return


def _validate_requirements_path(requirements_path: Path):
    requirements_path.resolve()
    if not requirements_path.exists():
        typer.echo(f"Could not find file: {requirements_path}")
        raise typer.Exit(1)
    if requirements_path.suffix not in {".txt", ".yml", ".yaml"}:
        typer.echo(
            "Requirements file in unexpected format. "
            f"Expected one of: .txt, .yml, .yaml; got {requirements_path.name}. "
        )
        raise typer.Exit(1)
    return
