import datetime
import logging
import shutil
import webbrowser
from pathlib import Path
from typing import Optional, cast
import json
import time

import docker  # type: ignore
import typer

from garden_ai import GardenClient, GardenConstants
from garden_ai.app.console import print_err
from garden_ai.containers import (
    build_image_with_dependencies,
    build_notebook_session_image,
    push_image_to_public_repo,
    start_container_with_notebook,
    get_docker_client,
    DockerStartFailure,
)
from garden_ai.local_data import _get_notebook_base_image, _put_notebook_base_image
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

    # We're most likely to see this error raised from get_docker_client,
    def __enter__(self):
        try:
            return get_docker_client()
        except DockerStartFailure as e:
            self.handle_docker_start_failure(e)

    # but if the user's Docker daemon shuts down partway through the session
    # we'll catch that here.
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is DockerStartFailure:
            self.handle_docker_start_failure(exc_val)


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
            "Path to a requirements.txt or a conda environment.yml containing "
            "additional dependencies to install in the base image."
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
    _put_notebook_base_image(notebook_path, base_image_uri)

    # Now we have all we need to prompt the user to proceed
    if need_to_create_notebook:
        message = f"This will create a new notebook {notebook_path.name} and open it in Docker image {base_image_uri}. "
    else:
        message = f"This will open existing notebook {notebook_path.name} in Docker image {base_image_uri}. "

    if requirements_path:
        message += f"Additional dependencies specified in {requirements_path.name} will also be installed in {base_image_uri}. "
        # sanity check requirements
        _validate_requirements_path(requirements_path)

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

    print(
        f"Starting notebook inside base image with full name {base_image_uri}. "
        f"If you start this notebook again from the same folder, it will use this image by default."
    )

    with DockerClientSession() as docker_client:
        # pre-bake local image with garden-ai and additional user requirements
        local_base_image_id = build_image_with_dependencies(
            docker_client, base_image_uri, requirements_path
        )
        # start container and listen for Ctrl-C
        container = start_container_with_notebook(
            docker_client, notebook_path, local_base_image_id, pull=False
        )
        _register_container_sigint_handler(container)

    typer.echo(
        f"Notebook started! Opening http://127.0.0.1:8888/notebooks/{notebook_path.name} "
        "in your default browser (you may need to refresh the page)"
    )

    # Give the notebook server a few seconds to start up so that the user doesn't have to refresh manually
    time.sleep(3)
    webbrowser.open_new_tab(f"http://127.0.0.1:8888/notebooks/{notebook_path.name}")

    # stream logs from the container
    for line in container.logs(stream=True):
        print(line.decode("utf-8"), end="")

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

    if notebook_path:
        last_used_image_uri = _get_notebook_base_image(notebook_path)
    else:
        last_used_image_uri = None

    if not any([base_image_name, custom_image_uri, last_used_image_uri]):
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

    # last_used_image_uri is definitely non-None at this point
    last_used_image_uri = cast(str, last_used_image_uri)

    # 3: If the user didn't specify an image explicitly, use the last image they used for this notebook.
    return last_used_image_uri


def _register_container_sigint_handler(container: docker.models.containers.Container):
    """helper: ensure SIGINT/ Ctrl-C to our CLI stops a given container"""
    import signal

    def handler(signal, frame):
        typer.echo("Stopping notebook...")
        container.stop()
        return

    signal.signal(signal.SIGINT, handler)
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
            "Path to a requirements.txt or a conda environment.yml containing "
            "additional dependencies to install in the base image."
        ),
    ),
):
    """Open the debugging notebook in a pre-prepared container.

    Changes to the notebook file will NOT persist after the container shuts down.
    Quit the process with Ctrl-C or by shutting down jupyter from the browser.
    """

    with DockerClientSession() as docker_client:
        base_image = (
            _get_notebook_base_image(path) or "gardenai/base:python-3.10-jupyter"
        )

        if requirements_path is not None:
            _validate_requirements_path(requirements_path)

        local_base_image_id = build_image_with_dependencies(
            docker_client, base_image, requirements_path
        )

        image = build_notebook_session_image(
            docker_client, path, local_base_image_id, pull=False
        )
        if image is None:
            typer.echo("Failed to build image.")
            raise typer.Exit(1)
        image_name = str(image.id)  # str used to guarantee type-check

        top_level_dir = Path(__file__).parent.parent
        debug_path = top_level_dir / "notebook_templates" / "debug.ipynb"

        container = start_container_with_notebook(
            docker_client, debug_path, image_name, mount=False, pull=False
        )
        _register_container_sigint_handler(container)

    typer.echo(
        f"Notebook started! Opening http://127.0.0.1:8888/notebooks/{debug_path.name} "
        "in your default browser (you may need to refresh the page)"
    )
    webbrowser.open_new_tab(f"http://127.0.0.1:8888/notebooks/{debug_path.name}")

    # stream logs from the container
    for line in container.logs(stream=True):
        print(line.decode("utf-8"), end="")

    # block until the container finishes
    container.wait()
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
            "Path to a requirements.txt or a conda environment.yml containing "
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

    base_image_uri = _get_base_image_uri(
        base_image_name, custom_image_uri, notebook_path
    )
    _put_notebook_base_image(notebook_path, base_image_uri)
    print(f"Using base image: {base_image_uri}")

    # Pre-process the notebook and make sure it's not too big
    raw_notebook_contents = notebook_path.read_text()
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
    notebook_url = client.upload_notebook(notebook_contents, notebook_path.name)

    with DockerClientSession() as docker_client:
        # Build the image
        if requirements_path:
            _validate_requirements_path(requirements_path)

        local_base_image_id = build_image_with_dependencies(
            docker_client,
            base_image_uri,
            requirements_path,
            print_logs=verbose,
            pull=True,
        )
        image = build_notebook_session_image(
            docker_client,
            notebook_path,
            local_base_image_id,
            print_logs=verbose,
            pull=False,
        )
        if image is None:
            typer.echo("Failed to build image.")
            raise typer.Exit(1)
        typer.echo(f"Built image: {image}")

        # generate tag and push image to ECR
        auth_config = client._get_auth_config_for_ecr_push()

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        image_tag = f"{notebook_path.stem}-{timestamp}"

        typer.echo(f"Pushing image to repository: {GardenConstants.GARDEN_ECR_REPO}")
        full_image_location = push_image_to_public_repo(
            docker_client, image, image_tag, auth_config, print_logs=verbose
        )
        typer.echo(f"Successfully pushed image to: {full_image_location}")
        client._register_and_publish_from_user_image(
            docker_client,
            image,
            base_image_uri,
            full_image_location,
            notebook_url,
        )


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
