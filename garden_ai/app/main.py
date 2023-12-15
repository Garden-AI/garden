import logging
from pathlib import Path
from typing import Optional

import rich
import typer

from garden_ai.app.garden import garden_app
from garden_ai.app.entrypoint import entrypoint_app
from garden_ai.app.notebook import notebook_app

from garden_ai import GardenClient, GardenConstants
from garden_ai._version import __version__
from garden_ai.local_data import _get_user_email, _clear_user_email

logger = logging.getLogger()

app = typer.Typer(no_args_is_help=True)


# nb: subcommands are mini typer apps in their own right
app.add_typer(garden_app)
app.add_typer(entrypoint_app)
app.add_typer(notebook_app)


def show_version(show: bool):
    """Display the installed version and quit."""
    if show:
        rich.print(f"garden-ai {__version__}")
        raise typer.Exit()


@app.command()
def whoami():
    """Print the email of the currently logged in user. If logged out, attempt a login."""
    user = _get_user_email()
    if user != "unknown":
        rich.print(user)
    else:
        # attempt login, if necessary
        GardenClient()


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
    _clear_user_email()
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
