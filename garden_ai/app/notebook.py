import logging
import subprocess
import time

import typer

from pathlib import Path
from tempfile import TemporaryDirectory

from garden_ai import GardenClient
from garden_ai.utils._meta import redef_in_main
from garden_ai.app.console import console
from garden_ai.container.containerize import (  # type: ignore
    IMAGE_NAME,
    build_container,
    start_container,
)

logger = logging.getLogger()

notebook_app = typer.Typer(name="notebook")


@notebook_app.callback(invoke_without_command=True)
def notebook(ctx: typer.Context):
    """sub-commands for operating with IPython notebooks"""
    if ctx.invoked_subcommand is None:
        build_container(jupyter=True)
        start_container(jupyter=True)


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
        subprocess.run(["docker", "cp", req_file, f"{CONTAINER_NAME}:/garden"])

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
    subprocess.run(["docker", "exec", CONTAINER_NAME, "/bin/bash", "-c", "./plant.sh"])
    subprocess.run(["docker", "commit", CONTAINER_NAME, f"{IMAGE_NAME}-planted"])
    subprocess.run(["docker", "stop", CONTAINER_NAME])
    time.sleep(3)  # takes a moment for the container to fully stop
    subprocess.run(["docker", "rmi", image_name])


@notebook_app.command(no_args_is_help=True)
def debug(
    image_id: Path = typer.Argument(
        None,
        help=("ID for a local planted container image to debug."),
    )
):
    interpreter_cmd = 'python -i -c \'import dill; dill.load_session("session.pkl"); print("Your notebook state has been loaded!")\''
    start_container(image_id, entrypoint="/bin/bash", args=["-c", interpreter_cmd])


def _funcx_invoke_pipeline(*args, **kwargs):
    import subprocess

    # overwrite any init process that utilized an unsupported dill version
    subprocess.run(["python", "-m", "pip", "install", "dill==0.3.5.1"])

    import dill

    dill.load_session("session.pkl")

    for obj in globals().values():
        if getattr(obj, "__name__", None) == "garden_target" and getattr(
            obj, "_check", None
        ):
            target_func = obj
            break
    else:
        raise ValueError("No function marked for invocation.")

    return target_func(*args, **kwargs)


@notebook_app.command()
def publish(
    tag: Path = typer.Argument(
        None,
        help=("Tag for the publically reachable planted container image."),
    )
):
    # add container to docker registry (when updating the container, the name MUST be changed or the cache lookup will find old version)
    # subprocess.run(["docker", "push", "idarling/public:garden.vPy"])

    import __main__

    # perform container and function registration
    client = GardenClient()
    container_id = client.compute_client.register_container(
        f"docker.io/idarling/public:{tag}", "docker"
    )
    print(f"Your container has been registered with UUID: {container_id}")

    redef_in_main(_funcx_invoke_pipeline)

    func_id = client.compute_client.register_function(
        __main__._funcx_invoke_pipeline, container_uuid=container_id, public=True
    )
    print(f"Your function has been registered with UUID: {func_id}")
