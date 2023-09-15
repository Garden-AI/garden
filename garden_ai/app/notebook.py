import logging
import subprocess

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


@notebook_app.command()
def build():
    build_container(jupyter=True)


@notebook_app.command(no_args_is_help=True)
def bake(
    notebook: Path = typer.Argument(
        None,
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
        help=("Path to a .ipynb file containing your pipeline implementation."),
    )
):
    if not notebook.exists() or not notebook.is_file() or notebook.suffix != ".ipynb":
        console.log(
            f"{notebook} is not a valid notebook file. Please provide a valid notebook file (.ipynb)."
        )
        raise typer.Exit(code=1)

    CONTAINER_NAME = "garden_baking"
    image_name = f"{IMAGE_NAME}-jupyter"
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

    subprocess.run(["docker", "cp", notebook, f"{CONTAINER_NAME}:/garden"])
    with TemporaryDirectory() as tmpdir:
        tmpdir = tmpdir.replace("\\", "/")  # Windows compatibility

        with open(f"{tmpdir}/bake.sh", "w+") as script:
            script.writelines(
                [
                    f"jupyter nbconvert --to python /garden/{notebook.name}\n",
                    f"printf \"\nimport dill\ndill.dump_session('session.pkl')\n\" >> /garden/{notebook.stem}.py\n",
                    f"ipython /garden/{notebook.stem}.py",
                ]
            )

        subprocess.run(
            ["docker", "cp", f"{tmpdir}/bake.sh", f"{CONTAINER_NAME}:/garden"]
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
    subprocess.run(["docker", "exec", CONTAINER_NAME, "/bin/bash", "-c", "./bake.sh"])
    subprocess.run(["docker", "commit", CONTAINER_NAME, f"{IMAGE_NAME}-baked"])
    subprocess.run(["docker", "stop", CONTAINER_NAME])


def _funcx_invoke_pipeline(*args, **kwargs):
    import subprocess

    # overwrite any init process that utilized an unsupported dill version
    subprocess.run(["python", "-m", "pip", "install", "dill==0.3.5.1"])

    import dill

    dill.load_session("session.pkl")

    # don't ask how I figured it out, I won't say.
    # but apparently dill populates locals() in the interpreter and globals() when running a script
    # *shrug*
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
        help=("Tag for the publically reachable baked container image."),
    )
):
    # add container to docker registry
    # when updating the container, the name MUST be changed (or the cache lookup will find old version)
    # subprocess.run(["docker", "push", "idarling/public:garden.v2"])

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


@notebook_app.command(no_args_is_help=True)
def debug(
    image_id: Path = typer.Argument(
        None,
        help=("ID for a local baked container image to debug."),
    )
):
    interpreter_cmd = 'python -i -c \'import dill; dill.load_module("session.pkl"); print("Your notebook state has been loaded!")\''
    start_container(image_id, entrypoint="/bin/bash", args=["-c", interpreter_cmd])
