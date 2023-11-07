import logging
import platform
from typing import Optional

import docker
import typer

from garden_ai.app.console import console
from garden_ai.app.garden import garden_app
from garden_ai.app.pipeline import pipeline_app
from garden_ai.app.notebook import notebook_app
from garden_ai._version import __version__

logger = logging.getLogger()

app = typer.Typer(no_args_is_help=True)


# nb: subcommands are mini typer apps in their own right
app.add_typer(garden_app)
app.add_typer(pipeline_app)
app.add_typer(notebook_app)


def show_version(show: bool):
    """Display the installed version and quit."""
    if show:
        console.print(f"garden-ai {__version__}")
        raise typer.Exit()


@app.callback()
def main_info(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=show_version, is_eager=True
    )
):
    """
    🌱 Hello, Garden 🌱
    """
    pass


@app.command(no_args_is_help=True)
def setup(
    repo: str = typer.Argument(
        ...,
        help=("Name of a Repository you can push docker images to."),
    ),
):
    """Perform checks on the Docker Desktop configuration.
    Report all errors encountered with possible fixes."""
    checks_passed = True

    try:
        client = docker.from_env()
        res = client.login("gardenai")
        print(res)

    except docker.errors.DockerException:
        checks_passed = False
        console.print("[bold red]Setup found an issue with Docker Desktop!")
        console.print("[bold red]Please ensure you have Docker Desktop running.")

        # an issue Max discovered on Mac systems (#318)
        if platform.system() == "Darwin":
            console.print(
                "[bold red]And please ensure the toggle is set to 'on' at "
                "Settings > Advanced > Allow the default Docker socket to be used."
            )

    if checks_passed:
        # the double space is needed to not overlap characters
        console.print("[bold green3]✔️  Setup passed all checks!")
