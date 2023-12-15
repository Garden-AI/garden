import logging
from typing import Optional

import rich
import typer

from garden_ai.app.garden import garden_app
from garden_ai.app.entrypoint import entrypoint_app
from garden_ai.app.notebook import notebook_app
from garden_ai import GardenClient
from garden_ai._version import __version__
from garden_ai.local_data import _get_user_email

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
    user = _get_user_email()
    if user != "unknown":
        rich.print(user)
    else:
        # undefined behavior if the user is not actually logged out
        GardenClient()


@app.command()
def login():
    if GardenClient().auth_key_store.file_exists():
        rich.print("Already logged in.")


@app.command()
def logout():
    # assumes user is logged in, seems like a safe assumption
    # otherwise: undefined behavior
    GardenClient().auth_key_store.clear_garden_data()


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
