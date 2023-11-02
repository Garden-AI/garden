import inspect
import json
import logging
import os
import shutil
import subprocess
import textwrap
import time
import webbrowser
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Dict, List, Optional
from uuid import UUID

import docker  # type: ignore
import typer

from garden_ai import GardenClient, GardenConstants, RegisteredPipeline, local_data
from garden_ai.app.console import console
from garden_ai.container.containerize import (  # type: ignore
    IMAGE_NAME,
    build_container,
    start_container,
)
from garden_ai.containers import (
    JUPYTER_TOKEN,
    build_notebook_session_image,
    extract_metadata_from_image,
    start_container_with_notebook,
)
from garden_ai.local_data import _get_notebook_base_image, _put_notebook_base_image
from garden_ai.utils._meta import redef_in_main

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
def plant(
    path: Path = typer.Argument(
        ...,
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
    ),
    base_image: Optional[str] = typer.Option(None),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    notebook_path = path.resolve()
    if notebook_path.suffix != ".ipynb":
        raise ValueError("File must be a jupyter notebook (.ipynb)")
    if not notebook_path.exists():
        raise ValueError(f"Could not find file at {notebook_path}")

    base_image = (
        base_image or _get_notebook_base_image(notebook_path) or "gardenai/test:latest"
    )
    _put_notebook_base_image(notebook_path, base_image)

    docker_client = docker.from_env()
    image = build_notebook_session_image(
        docker_client, notebook_path, base_image, print_logs=verbose
    )
    if image is None:
        typer.echo("Failed to build image.")
        raise typer.Exit(1)

    typer.echo(f"Built image: {image}")
    typer.echo("Extracting metadata ...\n")
    metadata = extract_metadata_from_image(docker_client, image)
    print(metadata)


def _make_function_to_register(func_name: str):
    def call_pipeline(*args, **kwargs):
        import dill  # type: ignore

        dill.load_session("session.pkl")
        func = globals()[func_name]
        return func(*args, **kwargs)

    return call_pipeline


def _funcx_invoke_pipeline(short_name: str, *args, **kwargs):
    import subprocess

    # overwrite any init process that utilized an unsupported dill version
    subprocess.run(["python", "-m", "pip", "install", "dill==0.3.5.1"])

    import dill  # type: ignore

    dill.load_session("session.pkl")

    for obj in globals().values():
        if (
            getattr(obj, "__name__", None) == "garden_target"
            and (meta := getattr(obj, "_pipeline_meta", None)) is not None
            and meta["short_name"] == short_name
        ):
            target_func = obj
            break
    else:
        raise ValueError(
            "No function matching the requested name was marked for invocation."
        )

    return target_func(*args, **kwargs)


def _curry(func, arg):
    def wrapper(*args, **kwargs):
        return func(arg, *args, **kwargs)

    return wrapper


@notebook_app.command(no_args_is_help=True)
def register(
    image: str = typer.Argument(
        ...,
        help=("The name of the image to be pushed and registered with Globus compute."),
    )
):
    # add container to docker registry
    # when updating the container, the name MUST be changed or the cache lookup will find old version
    # TODO
    subprocess.run(["docker", "tag", f"{IMAGE_NAME}-planted", image])
    subprocess.run(["docker", "push", image])
    subprocess.run(["docker", "rmi", image])

    client = GardenClient()

    container_id = client.compute_client.register_container(
        f"docker.io/{image}", "docker"
    )
    container_uuid = UUID(container_id)

    # extract metadata
    docker_client = docker.from_env()
    container = docker_client.containers.run(
        image=image,
        entrypoint="/bin/sh",
        command=["-c", "cat /garden/metadata.json"],
        remove=True,
        detach=False,
    )
    raw_metadata = container.decode("utf-8")

    # register pipelines from metadata
    total_meta = json.loads(raw_metadata)

    for key, record in total_meta.items():
        if "." in key:  # ignore connectors metadata
            continue
        if record["doi"] is None:
            record["doi"] = client._mint_draft_doi()
        record["container_uuid"] = container_uuid
        # n.b. key is the original function name
        to_register = _make_function_to_register(key)
        record["func_uuid"] = client.compute_client.register_function(
            to_register, container_uuid=container_id, public=True
        )
        registered = RegisteredPipeline(**record)
        client._update_datacite(registered)
        local_data.put_local_pipeline(registered)
        print(f"Successfully registered pipeline: {key}: {registered.doi}")
        # TODO publish to a garden too
