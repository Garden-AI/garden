"""CLI commands for Modal function and app management."""

import ast
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import rich
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from garden_ai import GardenClient
from garden_ai.app.utils import parse_list
from garden_ai.schemas.modal_app import (
    AsyncModalJobStatus,
    ModalAppCreateRequest,
    ModalFunctionCreateMetadata,
    ModalFunctionPatchRequest,
)

modal_app = typer.Typer(help="Manage Modal functions and apps", no_args_is_help=True)
modal_app_app = typer.Typer(help="Manage Modal apps", no_args_is_help=True)
modal_app.add_typer(modal_app_app, name="app")


def _extract_functions_from_file(file_path: Path) -> list[dict]:
    """Extract function metadata from a Modal Python file.

    This is a basic parser that looks for decorated functions.
    In practice, the backend does more sophisticated parsing.
    """
    content = file_path.read_text()
    tree = ast.parse(content)

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check if decorated with @app.function or similar
            for decorator in node.decorator_list:
                dec_str = ast.unparse(decorator) if hasattr(ast, "unparse") else ""
                if "function" in dec_str.lower() or "method" in dec_str.lower():
                    # Get docstring
                    docstring = ast.get_docstring(node) or ""

                    functions.append(
                        {
                            "function_name": node.name,
                            "docstring": docstring,
                            "lineno": node.lineno,
                        }
                    )
                    break

    return functions


def _extract_app_name(file_path: Path) -> str | None:
    """Try to extract the Modal app name from the file."""
    content = file_path.read_text()
    tree = ast.parse(content)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "app":
                    # Look for modal.App("name") pattern
                    if isinstance(node.value, ast.Call):
                        if node.value.args:
                            arg = node.value.args[0]
                            if isinstance(arg, ast.Constant):
                                return arg.value
    return None


# =============================================================================
# Modal Function Commands
# =============================================================================


@modal_app.command("list")
def list_modal_functions(
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum results"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON output"),
):
    """List your Modal functions."""
    client = GardenClient()
    functions = client.backend_client.get_modal_functions(limit=limit)

    if json_output:
        data = [fn.model_dump(mode="json") for fn in functions]
        if pretty:
            rich.print(data)
        else:
            print(json.dumps(data))
        return

    if not functions:
        rich.print("[yellow]No Modal functions found.[/yellow]")
        return

    table = Table(title="Modal Functions")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Title")
    table.add_column("App ID")
    table.add_column("Authors")
    table.add_column("Invocations", justify="right")

    for fn in functions:
        title = fn.title or "-"
        title_display = (title[:25] + "...") if len(title) > 25 else title
        authors = ", ".join(fn.authors[:2]) if fn.authors else "-"
        if fn.authors and len(fn.authors) > 2:
            authors += "..."

        table.add_row(
            str(fn.id),
            fn.function_name or "-",
            title_display,
            str(fn.modal_app_id) if fn.modal_app_id else "-",
            authors,
            str(fn.num_invocations),
        )

    rich.print(table)


@modal_app.command("show")
def show_modal_function(
    function_id: int = typer.Argument(..., help="Modal function ID"),
    show_code: bool = typer.Option(False, "--code", "-c", help="Show function code"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON output"),
):
    """Show details of a Modal function."""
    client = GardenClient()
    fn = client.backend_client.get_modal_function(function_id)

    if json_output:
        data = fn.model_dump(mode="json")
        if pretty:
            rich.print(data)
        else:
            print(json.dumps(data))
        return

    rich.print(f"\n[bold]Modal Function: {fn.function_name}[/bold]")
    rich.print(f"  ID: [cyan]{fn.id}[/cyan]")
    rich.print(f"  App ID: {fn.modal_app_id}")
    if fn.doi:
        rich.print(f"  DOI: {fn.doi}")
    rich.print(f"  Title: {fn.title}")
    if fn.description:
        rich.print(f"  Description: {fn.description}")
    if fn.authors:
        rich.print(f"  Authors: {', '.join(fn.authors)}")
    if fn.tags:
        rich.print(f"  Tags: {', '.join(fn.tags)}")
    rich.print(f"  Year: {fn.year}")
    rich.print(f"  Invocations: {fn.num_invocations}")
    rich.print(f"  Archived: {fn.is_archived}")

    if show_code and fn.function_text:
        rich.print("\n  [bold]Function Code:[/bold]")
        from rich.syntax import Syntax

        syntax = Syntax(fn.function_text, "python", theme="monokai", line_numbers=True)
        rich.print(syntax)


@modal_app.command("update")
def update_modal_function(
    function_id: int = typer.Argument(..., help="Modal function ID"),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="New title"),
    description: Optional[str] = typer.Option(
        None, "--description", "-d", help="New description"
    ),
    authors: Optional[str] = typer.Option(
        None, "--authors", "-a", help="New comma-separated authors"
    ),
    tags: Optional[str] = typer.Option(None, "--tags", help="New comma-separated tags"),
):
    """Update a Modal function's metadata."""
    client = GardenClient()

    request = ModalFunctionPatchRequest(
        title=title,
        description=description,
        authors=parse_list(authors) if authors else None,
        tags=parse_list(tags) if tags else None,
    )

    if not any([request.title, request.description, request.authors, request.tags]):
        rich.print("[yellow]No updates specified.[/yellow]")
        raise typer.Exit(1)

    fn = client.backend_client.patch_modal_function(function_id, request)

    rich.print("[green]✓[/green] Function updated successfully!")
    rich.print(f"  ID: {fn.id}")
    rich.print(f"  Name: {fn.function_name}")


# =============================================================================
# Modal App Commands
# =============================================================================


@modal_app_app.command("deploy")
def deploy_modal_app(
    file: Path = typer.Argument(..., help="Path to Modal Python file"),
    app_name: Optional[str] = typer.Option(
        None, "--name", "-n", help="App name (auto-detected if not provided)"
    ),
    title: Optional[str] = typer.Option(
        None, "--title", "-t", help="Title for functions (defaults to app name)"
    ),
    authors: Optional[str] = typer.Option(
        None, "--authors", "-a", help="Comma-separated list of authors"
    ),
    tags: Optional[str] = typer.Option(
        None, "--tags", help="Comma-separated list of tags"
    ),
    base_image: str = typer.Option(
        "python:3.11-slim", "--base-image", help="Base Docker image"
    ),
    requirements: Optional[str] = typer.Option(
        None, "--requirements", "-r", help="Comma-separated pip requirements"
    ),
    wait: bool = typer.Option(
        True, "--wait/--no-wait", help="Wait for deployment to complete"
    ),
    timeout: float = typer.Option(
        300.0, "--timeout", help="Deployment timeout in seconds"
    ),
):
    """Deploy a Modal app from a Python file."""
    if not file.exists():
        rich.print(f"[red]Error:[/red] File not found: {file}")
        raise typer.Exit(1)

    client = GardenClient()
    file_contents = file.read_text()

    # Try to extract app name from file if not provided
    if not app_name:
        app_name = _extract_app_name(file) or file.stem
        rich.print(f"[dim]Using app name: {app_name}[/dim]")

    # Extract function info from file
    functions_info = _extract_functions_from_file(file)
    if not functions_info:
        rich.print("[yellow]Warning:[/yellow] No decorated functions found in file.")
        rich.print("Make sure your functions are decorated with @app.function()")

    # Build function metadata
    year = str(datetime.now().year)
    author_list = parse_list(authors) or [client.get_email()]
    tag_list = parse_list(tags)

    modal_functions = []
    for fn_info in functions_info:
        modal_functions.append(
            ModalFunctionCreateMetadata(
                function_name=fn_info["function_name"],
                title=title or fn_info["function_name"],
                description=fn_info["docstring"] or None,
                year=year,
                authors=author_list,
                tags=tag_list,
                function_text=file_contents,  # Full file for now
            )
        )

    if not modal_functions:
        # If we couldn't parse functions, create a placeholder
        rich.print(
            "[yellow]Could not auto-detect functions. "
            "Backend will parse the file.[/yellow]"
        )

    request = ModalAppCreateRequest(
        app_name=app_name,
        file_contents=file_contents,
        base_image_name=base_image,
        requirements=parse_list(requirements),
        modal_functions=modal_functions,
    )

    rich.print(f"\n[bold]Deploying Modal app:[/bold] {app_name}")

    if wait:
        # Use async endpoint and poll
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task("Deploying...", total=None)

            app_response = client.backend_client.create_modal_app_async(request)
            progress.update(
                task, description=f"Deployment started (ID: {app_response.id})"
            )

            try:
                app_response = client.backend_client.poll_modal_app_deployment(
                    app_response.id, timeout=timeout
                )
            except TimeoutError as e:
                rich.print(f"[red]Error:[/red] {e}")
                raise typer.Exit(1)

        if app_response.deploy_status == AsyncModalJobStatus.FAILED:
            rich.print("[red]✗ Deployment failed![/red]")
            if app_response.deploy_error:
                rich.print(f"  Error: {app_response.deploy_error}")
            if app_response.suggested_fix:
                rich.print(f"  Suggested fix: {app_response.suggested_fix}")
            raise typer.Exit(1)
    else:
        app_response = client.backend_client.create_modal_app_async(request)

    rich.print("\n[green]✓[/green] Modal app deployed successfully!")
    rich.print(f"  App ID: [bold]{app_response.id}[/bold]")
    rich.print(f"  App Name: {app_response.app_name}")
    rich.print(f"  Status: {app_response.deploy_status or 'pending'}")

    if app_response.modal_functions:
        rich.print("\n  [bold]Functions:[/bold]")
        for fn in app_response.modal_functions:
            rich.print(f"    - {fn.function_name} (ID: {fn.id})")


@modal_app_app.command("list")
def list_modal_apps(
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON output"),
):
    """List your Modal apps."""
    client = GardenClient()
    apps = client.backend_client.get_modal_apps()

    if json_output:
        data = [app.model_dump(mode="json") for app in apps]
        if pretty:
            rich.print(data)
        else:
            print(json.dumps(data))
        return

    if not apps:
        rich.print("[yellow]No Modal apps found.[/yellow]")
        return

    table = Table(title="Modal Apps")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Functions")

    for app in apps:
        status = app.deploy_status.value if app.deploy_status else "unknown"
        status_style = {
            "success": "green",
            "failed": "red",
            "pending": "yellow",
            "running": "yellow",
        }.get(status, "dim")

        table.add_row(
            str(app.id),
            app.original_app_name or app.app_name,
            f"[{status_style}]{status}[/{status_style}]",
            str(len(app.modal_functions)),
        )

    rich.print(table)


@modal_app_app.command("show")
def show_modal_app(
    app_id: int = typer.Argument(..., help="Modal app ID"),
    show_code: bool = typer.Option(False, "--code", "-c", help="Show file contents"),
    show_app_text: bool = typer.Option(
        False, "--show-app-text", help="Show the app_text field (deployed code)"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON output"),
):
    """Show details of a Modal app."""
    client = GardenClient()
    app = client.backend_client.get_modal_app(app_id)

    if json_output:
        data = app.model_dump(mode="json")
        if pretty:
            rich.print(data)
        else:
            print(json.dumps(data))
        return

    status = app.deploy_status.value if app.deploy_status else "unknown"
    status_style = {"success": "green", "failed": "red"}.get(status, "yellow")

    rich.print(f"\n[bold]Modal App: {app.original_app_name or app.app_name}[/bold]")
    rich.print(f"  ID: [cyan]{app.id}[/cyan]")
    rich.print(f"  Status: [{status_style}]{status}[/{status_style}]")
    rich.print(f"  Base Image: {app.base_image_name}")

    if app.requirements:
        rich.print(f"  Requirements: {', '.join(app.requirements)}")

    if app.deploy_error:
        rich.print(f"  [red]Error:[/red] {app.deploy_error}")
    if app.suggested_fix:
        rich.print(f"  [yellow]Fix:[/yellow] {app.suggested_fix}")

    if app.modal_functions:
        rich.print(f"\n  [bold]Functions ({len(app.modal_functions)}):[/bold]")
        for fn in app.modal_functions:
            doi_str = f" [DOI: {fn.doi}]" if fn.doi else ""
            rich.print(f"    - {fn.function_name} (ID: {fn.id}){doi_str}")
            if fn.title and fn.title != fn.function_name:
                rich.print(f"      Title: {fn.title}")

    if show_code and app.file_contents:
        rich.print("\n  [bold]File Contents:[/bold]")
        from rich.syntax import Syntax

        syntax = Syntax(app.file_contents, "python", theme="monokai", line_numbers=True)
        rich.print(syntax)

    if show_app_text and hasattr(app, "app_text") and app.app_text:
        rich.print("\n  [bold]App Text (Deployed Code):[/bold]")
        from rich.syntax import Syntax

        syntax = Syntax(app.app_text, "python", theme="monokai", line_numbers=True)
        rich.print(syntax)


@modal_app_app.command("delete")
def delete_modal_app(
    app_id: int = typer.Argument(..., help="Modal app ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete a Modal app and its functions."""
    client = GardenClient()

    # Get app details first
    app = client.backend_client.get_modal_app(app_id)

    if not force:
        msg = f"Delete Modal app '{app.original_app_name or app.app_name}' with {len(app.modal_functions)} function(s)?"
        if not typer.confirm(msg):
            rich.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    client.backend_client.delete_modal_app(app_id)
    rich.print(f"[green]✓[/green] Modal app {app_id} deleted successfully!")
