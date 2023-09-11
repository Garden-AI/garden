import logging
import subprocess

import typer

from pathlib import Path
from tempfile import TemporaryDirectory

from garden_ai.app.console import console
from garden_ai.container.containerize import (
    IMAGE_NAME,
    build_container,
    start_container,
)

logger = logging.getLogger()

notebook_app = typer.Typer(name="notebook")


@notebook_app.callback(invoke_without_command=True)
def notebook():
    """sub-commands for operating with IPython notebooks"""
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

    copy = subprocess.run(["docker", "cp", notebook, "gardenai/base-jupyter:/garden"])
    if copy.returncode:
        console.log(
            f"Copy operation failed. If the target container has not been built, please run `garden-ai notebook build`."
        )
        raise typer.Exit(code=1)

    with TemporaryDirectory() as tmpdir:
        tmpdir = tmpdir.replace("\\", "/")  # Windows compatibility

        with open(f"{tmpdir}/bake.sh", "w+") as script:
            script.writelines(
                [
                    f"jupyter nbconvert --to script /garden/{notebook.name}\n",
                    f'printf "\nimport dill\ndill.dump_module()" >> /garden/{notebook.stem}.py\n',  # saves to /tmp/session.pkl
                    f"ipython /garden/{notebook.stem}.py",
                ]
            )

        subprocess.run(
            ["docker", "cp", f"{tmpdir}/bake.sh", "gardenai/base-jupyter:/garden"]
        )

    start_container(
        container_name="garden_baking",
        entrypoint="/bin/bash",
        args=["/garden/bake.sh"],
    )
    subprocess.run(["docker", "commit", "garden_baking", f"{IMAGE_NAME}-baked"])


@notebook_app.command(no_args_is_help=True)
def debug(
    image_id: Path = typer.Argument(
        None,
        help=("ID for a local baked container image to debug."),
    )
):
    interpreter_cmd = [
        "python",
        "-i",
        "-c",
        f"'import dill; dill.load_module(); del dill; print(\"Your notebook state has been loaded!\")'",  # loads from /tmp/session.pkl
    ]
    start_container(image_id, entrypoint="/bin/bash", args=interpreter_cmd)
