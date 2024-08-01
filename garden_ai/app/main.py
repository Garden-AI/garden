import logging
import os
from pathlib import Path
from typing import Optional

import rich
import typer

from garden_ai import GardenClient, GardenConstants
from garden_ai._version import __version__
from garden_ai.app.docker import docker_app
from garden_ai.app.entrypoint import entrypoint_app
from garden_ai.app.garden import garden_app
from garden_ai.app.hpc_notebook import hpc_notebook_app
from garden_ai.app.notebook import notebook_app

logger = logging.getLogger()

app = typer.Typer(no_args_is_help=True)


# nb: subcommands are mini typer apps in their own right
app.add_typer(garden_app)
app.add_typer(entrypoint_app)
app.add_typer(notebook_app)
app.add_typer(docker_app)
app.add_typer(hpc_notebook_app, hidden=True)


def show_version(show: bool):
    """Display the installed version and quit."""
    version_str = f"garden-ai {__version__}"
    if show:
        if env := os.environ.get("GARDEN_ENV"):
            version_str += f" ({env})"
        rich.print(version_str)
        raise typer.Exit()


@app.command()
def whoami():
    """Print the email of the currently logged in user. If logged out, attempt a login."""
    client = GardenClient()
    user = client.get_email()
    rich.print(user)


@app.command()
def login():
    """Attempts to login if the user is currently logged out."""
    if Path(GardenConstants.GARDEN_KEY_STORE).exists():
        rich.print("Already logged in.")
    else:
        # attempt login
        GardenClient()


@app.command()
def logout():
    """Logs out the current user."""
    # silently ignores the case where the file is already gone
    Path.unlink(Path(GardenConstants.GARDEN_KEY_STORE), missing_ok=True)


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
