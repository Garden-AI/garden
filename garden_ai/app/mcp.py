from typing import Annotated

import rich
import typer

from garden_ai.mcp.config_add import MCPConfigInitalizer as Init

mcp_app = typer.Typer(help="MCP server commands", no_args_is_help=True)


@mcp_app.callback()
def mcp():
    """
    sub-commands for local mcp server
    """
    pass


@mcp_app.command(no_args_is_help=True)
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
    if client and path:
        raise ValueError("Cannot specify both a client and a path")
    elif not client and not path:
        raise ValueError("Specify either a client or a path")

    if path:
        config_path = Init.setup_custom(path)
    elif client:
        match client.lower():
            case "claude":
                config_path = Init.setup_claude()
            case "claude-code":
                config_path = Init.setup_claude_code()
            case "gemini":
                config_path = Init.setup_gemini()
            case "cursor":
                config_path = Init.setup_cursor()
            case "windsurf":
                config_path = Init.setup_windsurf()
            case _:
                rich.print(
                    "Not supported for config initialization",
                    "Try 'claude', 'claude code', 'gemini', 'cursor', or 'windsurf' or specify a path to a config file",
                )
                return

    rich.print(f"Garden MCP configuration file set up at {config_path}")


@mcp_app.command(no_args_is_help=False)
def serve():
    """Start the Garden MCP server."""
    try:
        from garden_ai.mcp.server import main as mcp_main

        mcp_main()
    except ImportError:
        rich.print("[red]Error:[/red] MCP extra not installed.")
        rich.print("Install with: [cyan]pip install garden-ai[mcp][/cyan]")
        raise typer.Exit(1)
