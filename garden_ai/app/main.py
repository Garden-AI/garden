import logging
import os
from pathlib import Path
from typing import Optional, Annotated

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


mcp_app = typer.Typer(help="MCP server commands")
app.add_typer(mcp_app, name="mcp")


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


@mcp_app.command()
def setup(
    client: Annotated[
        str | None,
        typer.Option(help="'claude', 'claude code', 'gemini', 'cursor', 'windsurf'"),
    ] = None,
    path: Annotated[
        str | None,
        typer.Option(help="Path to initalize config file for any other mcp client"),
    ] = None,
):
    """Add config file for client"""
    from garden_ai.mcp.config_add import MCPConfigInitalizer as Init

    if client and path:
        raise ValueError("Cannot specify both a client and a path")
    elif not client and not path:
        raise ValueError("Specify either a client or a path")

    if path:
        config_path = Init.custom(path)
    else:
        match client.lower():
            case "claude":
                config_path = Init.claude()
            case "claude code":
                config_path = Init.claude_code()
            case "gemini":
                config_path = Init.gemini()
            case "cursor":
                config_path = Init.cursor()
            case "windsurf":
                config_path = Init.windsurf()
            case _:
                rich.print(
                    "Not supported for config initialization",
                    "Try 'claude', 'claude code', 'gemini', 'cursor', or 'windsurf' or specify a path to a config file",
                )
                return

    rich.print(f"Config file set up at {config_path}")


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
