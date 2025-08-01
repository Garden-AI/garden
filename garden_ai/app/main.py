import logging
import os
from pathlib import Path
from typing import Optional

import rich
import typer

from garden_ai import GardenClient, GardenConstants
from garden_ai._version import __version__

logger = logging.getLogger()

app = typer.Typer(no_args_is_help=True)


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


@app.command()
def mcp():
    """Start the Garden MCP server."""
    try:
        from garden_ai.mcp.server import main as mcp_main

        mcp_main()
    except ImportError:
        rich.print("[red]Error:[/red] MCP extra not installed.")
        rich.print("Install with: [cyan]pip install garden-ai[mcp][/cyan]")
        raise typer.Exit(1)


@app.callback()
def main_info(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=show_version, is_eager=True
    ),
):
    """
    🌱 Hello, Garden 🌱
    """
    pass
