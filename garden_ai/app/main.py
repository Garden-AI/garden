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


def user_operations(command: Optional[str]):
    """Handle whoami, login, or logout command then quit."""
    if command == "whoami":
        user = _get_user_email()
        if user != "unknown":
            rich.print(user)
        else:
            # undefined behavior if the user is not actually logged out
            GardenClient()
    elif command == "login":
        # forces the login flow when logged out
        GardenClient()
    elif command == "logout":
        # assumes user is logged in, seems like a safe assumption
        # otherwise: undefined behavior
        GardenClient().auth_key_store.clear_garden_data()
    else:
        if command is not None:
            rich.print(f"Unknown command: {command}.")
    # quit
    raise typer.Exit()


@app.callback()
def main_info(
    user_op: Optional[str] = typer.Argument(
        None, callback=user_operations, is_eager=True
    ),
    version: Optional[bool] = typer.Option(
        None, "--version", callback=show_version, is_eager=True
    ),
):
    """
    🌱 Hello, Garden 🌱
    """
    pass
