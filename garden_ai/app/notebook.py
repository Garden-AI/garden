import datetime
import logging
import shutil
import webbrowser
from pathlib import Path
from typing import Optional

import docker  # type: ignore
import typer

from garden_ai import GardenClient, GardenConstants, local_data
from garden_ai.containers import (
    JUPYTER_TOKEN,
    build_notebook_session_image,
    push_image_to_public_repo,
    start_container_with_notebook,
)
from garden_ai.local_data import _get_notebook_base_image, _put_notebook_base_image

logger = logging.getLogger()

notebook_app = typer.Typer(name="notebook")


@notebook_app.callback(no_args_is_help=True)
def notebook():
    """sub-commands for editing and publishing from sandboxed notebooks."""
    pass


@notebook_app.command()
def list_premade_images():
    """List all Garden base docker images"""
    premade_images = ", ".join(
        [
            "'" + image_name + "'"
            for image_name in list(GardenConstants.PREMADE_IMAGES.keys())
        ]
    )
    print(f"Garden premade images:\n{premade_images}")


@notebook_app.command(no_args_is_help=True)
def start(
    path: Path = typer.Argument(
        ...,
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
        help=("Path to a .ipynb notebook to open in a fresh, isolated container. "),
    ),
    base_image: Optional[str] = typer.Option(
        default=None,
        help=(
            "A Garden base image to boot the notebook in. "
            "For example, to boot your notebook with the default Garden python 3.8 image, use --base-image 3.8-base. "
            "To see all the available Garden base images, use 'garden-ai notebook list-premade-images'"
        ),
    ),
):
    """Open a notebook file in a sandboxed environment. Optionally, specify a different base docker image.

    Changes to the notebook file will persist after the container shuts down.
    Quit the process with Ctrl-C or by shutting down jupyter from the browser.
    If a different base image is chosen, that image will be reused as the default for this notebook in the future.
    """
    notebook_path = path.resolve()
    if notebook_path.suffix != ".ipynb":
        raise ValueError("File must be a jupyter notebook (.ipynb)")

    if not notebook_path.exists():
        top_level_dir = Path(__file__).parent.parent
        source_path = top_level_dir / "notebook_templates" / "sklearn.ipynb"
        shutil.copy(source_path, notebook_path)

    # check/update local data for base image choice
    if base_image in list(GardenConstants.PREMADE_IMAGES.keys()):
        base_image = GardenConstants.PREMADE_IMAGES[base_image]
    else:
        premade_images = ", ".join(
            [
                "'" + image_name + "'"
                for image_name in list(GardenConstants.PREMADE_IMAGES.keys())
            ]
        )
        raise Exception(
            f"The image '{base_image}' is not one of the Garen base images. The current Garden base images are: \n{premade_images}"
        )

    base_image = (
        base_image or _get_notebook_base_image(notebook_path) or "gardenai/test:latest"
    )
    _put_notebook_base_image(notebook_path, base_image)
    print(f"Using base image: {base_image}")

    # start container and listen for Ctrl-C
    docker_client = docker.from_env()
    container = start_container_with_notebook(docker_client, notebook_path, base_image)
    _register_container_sigint_handler(container)

    typer.echo(
        f"Notebook started! Opening http://127.0.0.1:8888/tree?token={JUPYTER_TOKEN} in your default browser (you may need to refresh the page)"
    )
    webbrowser.open_new_tab(f"http://127.0.0.1:8888/tree?token={JUPYTER_TOKEN}")

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


def _register_container_sigint_handler(container: docker.models.containers.Container):
    """helper: ensure SIGINT/ Ctrl-C to our CLI stops a given container"""
    import signal

    def handler(signal, frame):
        typer.echo("Stopping notebook...")
        container.stop()
        return

    signal.signal(signal.SIGINT, handler)
    return


@notebook_app.command()
def publish(
    path: Path = typer.Argument(
        ...,
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
    ),
    base_image: Optional[str] = typer.Option(None),
    image_repo: Optional[str] = typer.Option(
        None,
        "--repo",
        help=(
            "Name of a public Dockerhub repository to publish garden-generated "
            "images, e.g. `user/garden-images`. The repository must already "
            "exist and you must have push access to the repository. "
        ),
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    client = GardenClient()
    notebook_path = path.resolve()
    if notebook_path.suffix != ".ipynb":
        raise ValueError("File must be a jupyter notebook (.ipynb)")
    if not notebook_path.exists():
        raise ValueError(f"Could not find file at {notebook_path}")

    # check for preferred base image
    base_image = (
        base_image or _get_notebook_base_image(notebook_path) or "gardenai/test:latest"
    )
    _put_notebook_base_image(notebook_path, base_image)
    print(f"Using base image: {base_image}")

    # check for preferred image repository
    image_repo = image_repo or local_data._get_user_image_repo()

    if image_repo is None:
        raise ValueError("No image repository specified.")
    else:
        # remember for next time
        local_data._store_user_image_repo(image_repo)
    print(f"Using image repository: {image_repo}")

    # Build the image
    docker_client = docker.from_env()
    image = build_notebook_session_image(
        docker_client, notebook_path, base_image, print_logs=verbose
    )
    if image is None:
        typer.echo("Failed to build image.")
        raise typer.Exit(1)
    typer.echo(f"Built image: {image}")

    # generate tag and and push image to dockerhub
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    image_tag = f"{notebook_path.stem}-{timestamp}"

    typer.echo(f"Pushing image to repository: {image_repo}")
    image_location = push_image_to_public_repo(
        docker_client, image, image_repo, image_tag, print_logs=verbose
    )
    typer.echo(f"Successfully pushed image to: {image_location}")
    # register container and pipelines with globus compute; re-publish gardens
    client._register_and_publish_from_user_image(docker_client, image, image_location)
