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

from garden_ai import GardenClient, RegisteredPipeline, local_data
from garden_ai.app.console import console
from garden_ai.container.containerize import (  # type: ignore
    IMAGE_NAME,
    build_container,
    start_container,
)
from garden_ai.containers import JUPYTER_TOKEN, start_container_with_notebook
from garden_ai.local_data import _get_notebook_base_image, _put_notebook_base_image
from garden_ai.utils._meta import redef_in_main
from garden_ai import GardenConstants

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
        help=("Path to a .ipynb notebook to open in a fresh, isolated container."),
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
    container = start_container_with_notebook(notebook_path, docker_client, base_image)
    _register_container_sigint_handler(container)

    typer.echo(
        f"Notebook started! Opening http://127.0.0.1:8888/tree?token={JUPYTER_TOKEN} in your default browser (you may need to refresh the page)"
    )
    webbrowser.open_new_tab(f"http://127.0.0.1:8888/tree?token={JUPYTER_TOKEN}")

    # stream logs from the container
    for line in container.logs(stream=True):
        print(line.decode("utf-8"), end="")

    # block until the container finishes
    container.wait()
    typer.echo("Notebook has stopped.")
    return


def _register_container_sigint_handler(container: docker.models.containers.Container):
    """helper: ensure SIGINT/ Ctrl-C to our CLI stops a given container"""
    import signal

    def handler(signal, frame):
        """make SIGINT / Ctrl-C stop the container"""
        typer.echo("Stopping notebook...")
        container.stop()
        raise typer.Exit(0)

    signal.signal(signal.SIGINT, handler)
    return


@notebook_app.command(no_args_is_help=True)
def plant(
    source: Path = typer.Argument(
        ...,
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
        help=(
            "Path to a notebook or Python file containing your pipeline implementation."
        ),
    ),
    req_file: Path = typer.Option(
        None,
        "-r",
        "--requirements",
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
        help=("Path to your requirements.txt file, defaults to next to source file."),
    ),
    cp: List[Path] = typer.Option(
        None,
        "-c",
        "--copy",
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
        help=(
            "Path to any other file to be copied into the container. This option can be supplied multiple times."
        ),
    ),
):
    if (
        not source.exists()
        or not source.is_file()
        or source.suffix not in (".ipynb", ".py")
    ):
        console.log(
            f"{source} is not a valid notebook or Python file. Please provide a valid file (.ipynb/.py)."
        )
        raise typer.Exit(code=1)
    elif source.suffix == ".ipynb":
        jupyter = True
    else:
        jupyter = False

    CONTAINER_NAME = "garden_planting"
    image_name = f"{IMAGE_NAME}-planting"
    build_container(image_name, jupyter=jupyter)
    subprocess.run(
        [
            "docker",
            "run",
            "-it",
            "-d",
            "--rm",
            "--platform",
            "linux/x86_64",
            "--name",
            CONTAINER_NAME,
            "--entrypoint",
            "/bin/bash",
            image_name,
        ]
    )

    if not jupyter:
        if req_file is None:
            req_file = source.parent / "requirements.txt"

        if not req_file.exists() or not req_file.is_file():
            console.log(f"{req_file} does not appear to be a valid path.")
            raise typer.Exit(code=1)

        subprocess.run(["docker", "cp", req_file, f"{CONTAINER_NAME}:/garden"])

    if cp is not None:
        for path in cp:
            if not path.exists() or not path.is_file():
                console.log(f"{path} does not appear to be a valid path.")
                raise typer.Exit(code=1)
            subprocess.run(["docker", "cp", path, f"{CONTAINER_NAME}:/garden"])

    subprocess.run(["docker", "cp", source, f"{CONTAINER_NAME}:/garden"])
    with TemporaryDirectory() as tmpdir:
        tmpdir = tmpdir.replace("\\", "/")  # Windows compatibility

        with open(f"{tmpdir}/plant.sh", "w") as script:
            script.writelines(
                [
                    f"jupyter nbconvert --to python /garden/{source.name}\n"
                    if jupyter
                    else "python -m pip install -r /garden/requirements.txt\n",
                    f"printf \"\nimport dill\ndill.dump_session('session.pkl')\n\" >> /garden/{source.stem}.py\n",
                    f"{'i' if jupyter else ''}python /garden/{source.stem}.py",
                ]
            )

        subprocess.run(
            ["docker", "cp", f"{tmpdir}/plant.sh", f"{CONTAINER_NAME}:/garden"]
        )

    subprocess.run(
        [
            "docker",
            "exec",
            CONTAINER_NAME,
            "/bin/bash",
            "-c",
            "find . -type f -print0 | xargs -0 dos2unix",
        ]
    )
    subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "/bin/bash", "-c", "chmod +x plant.sh"]
    )
    subprocess.run(["docker", "exec", CONTAINER_NAME, "/bin/bash", "-c", "./plant.sh"])
    subprocess.run(["docker", "commit", CONTAINER_NAME, f"{IMAGE_NAME}-planted"])
    subprocess.run(["docker", "stop", CONTAINER_NAME])
    time.sleep(3)  # takes a moment for the container to fully stop
    subprocess.run(["docker", "rmi", image_name])


@notebook_app.command()
def debug(
    image: str = typer.Argument(
        f"{IMAGE_NAME}-planted",
        help=("Name/ID for a local planted container image to debug."),
    )
):
    """Open the debugging notebook in a pre-prepared container.

    Changes to the notebook file will NOT persist after the container shuts down.
    Quit the process with Ctrl-C or by shutting down jupyter from the browser.
    """
    top_level_dir = Path(__file__).parent.parent
    debug_path = top_level_dir / "notebook_templates" / "debug.ipynb"

    docker_client = docker.from_env()
    container = start_container_with_notebook(debug_path, docker_client, image, False)
    _register_container_sigint_handler(container)

    typer.echo(
        f"Notebook started! Opening http://127.0.0.1:8888/notebooks/debug.ipynb?token={JUPYTER_TOKEN} "
        "in your default browser (you may need to refresh the page)"
    )
    webbrowser.open_new_tab(
        f"http://127.0.0.1:8888/notebooks/debug.ipynb?token={JUPYTER_TOKEN}"
    )

    # stream logs from the container
    for line in container.logs(stream=True):
        print(line.decode("utf-8"), end="")

    # block until the container finishes
    container.wait()
    typer.echo("Notebook has stopped.")
    return


def _send_def_to_tmp_script(func) -> str:
    function_source = inspect.getsource(func)
    function_body = function_source.split(":", 1)[1].strip()
    # leading 4 spaces is because the first line wasn't being indented
    function_body_dedented = textwrap.dedent("    " + function_body)

    with NamedTemporaryFile(delete=False, suffix=".py") as f:
        f.write(function_body_dedented.encode())
        temp_file_name = f.name

    # caller must remove the returned file
    return temp_file_name


# Function to be run in planted container context.
# Sends model metadata up to host as printed JSON.
def _extract_metadata_from_planted_container() -> None:
    import json

    import dill  # type: ignore

    dill.load_session("session.pkl")

    garden_decorated = []
    global_vars = list(globals().values())

    for obj in global_vars:
        if getattr(obj, "__name__", None) == "garden_target" and hasattr(
            obj, "_pipeline_meta"  # could also check for _model_connectors
        ):
            garden_decorated.append(obj)

    if len(garden_decorated) == 0:
        raise ValueError("No functions marked with garden decorator.")

    total_meta = {}

    for marked in garden_decorated:
        key_name = marked._pipeline_meta["short_name"]
        connector_key = f"{key_name}.connectors"

        total_meta[key_name] = marked._pipeline_meta
        total_meta[connector_key] = [
            connector.metadata for connector in marked._model_connectors
        ]

    print(json.dumps(total_meta))  # stdout is captured


def _extract(planted_image: str) -> Dict[str, Any]:
    temp_file_name = _send_def_to_tmp_script(_extract_metadata_from_planted_container)
    mount_file = f"{temp_file_name}:/tmp/extract.py"

    stdout = start_container(
        planted_image,
        entrypoint="/bin/bash",
        mount_file=mount_file,
        args=["-c", "python /tmp/extract.py"],
    )

    # remove the temporary file
    os.remove(temp_file_name)

    return json.loads(stdout)


@notebook_app.command(no_args_is_help=True)
def extract(
    image: str = typer.Argument(
        ...,
        help=("The name of the planted image to extract metadata from."),
    ),
):
    print(_extract(image))


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
    subprocess.run(["docker", "tag", f"{IMAGE_NAME}-planted", image])
    subprocess.run(["docker", "push", image])
    subprocess.run(["docker", "rmi", image])

    client = GardenClient()

    container_id = client.compute_client.register_container(
        f"docker.io/{image}", "docker"
    )
    container_uuid = UUID(container_id)
    # print(f"Your container has been registered with UUID: {container_id}")

    pipeline_metas = []
    total_meta = _extract(image)
    for key, meta in total_meta.items():
        if "." in key:  # ignore connectors metadata
            continue
        if meta["doi"] is None:
            meta["doi"] = client._mint_draft_doi()
        pipeline_metas.append({"container_uuid": container_uuid, **meta})

    import __main__

    redef_in_main(_funcx_invoke_pipeline)

    funcx_unique_pipeline_funcs = [
        _curry(__main__._funcx_invoke_pipeline, pipeline["short_name"])
        for pipeline in pipeline_metas
    ]

    func_ids = [
        client.compute_client.register_function(
            unique_pipeline_func, container_uuid=container_id, public=True
        )
        for unique_pipeline_func in funcx_unique_pipeline_funcs
    ]
    # print(f"Your function(s) has (have) been registered with UUID(s): {func_ids}")

    registered_pipelines = []
    for i, pipeline in enumerate(pipeline_metas):
        pipeline["func_uuid"] = UUID(func_ids[i])

        registered = RegisteredPipeline(**pipeline)
        registered_pipelines.append(registered)

        client._update_datacite(registered)
        local_data.put_local_pipeline(registered)

    pipeline_name_to_doi = {
        registered.short_name: registered.doi for registered in registered_pipelines
    }

    print(
        f"Successfully registered your new pipeline(s) with doi(s): {pipeline_name_to_doi}"
    )
