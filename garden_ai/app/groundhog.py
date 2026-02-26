"""CLI commands for HPC (HPC) function management."""

import ast
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import rich
import typer
from rich.table import Table

from garden_ai import GardenClient
from garden_ai.app.utils import parse_int_list, parse_list
from garden_ai.schemas.groundhog import (
    HpcEndpointCreateRequest,
    HpcEndpointPatchRequest,
    HpcFunctionCreateRequest,
    HpcFunctionPatchRequest,
)

groundhog_app = typer.Typer(help="Manage HPC functions", no_args_is_help=True)
endpoint_app = typer.Typer(help="Manage HPC endpoints", no_args_is_help=True)
groundhog_app.add_typer(endpoint_app, name="endpoint")


def _has_hog_decorator(node: ast.FunctionDef) -> bool:
    """Check if a function/method has @hog.function or @hog.method decorator."""
    for decorator in node.decorator_list:
        # Handle @hog.function() or @hog.method() with parens
        if isinstance(decorator, ast.Call) and isinstance(
            decorator.func, ast.Attribute
        ):
            if (
                isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id == "hog"
                and decorator.func.attr in ("function", "method")
            ):
                return True
        # Handle @hog.function or @hog.method without parens
        if isinstance(decorator, ast.Attribute):
            if (
                isinstance(decorator.value, ast.Name)
                and decorator.value.id == "hog"
                and decorator.attr in ("function", "method")
            ):
                return True
    return False


def _extract_function_from_file(file_path: Path) -> dict:
    """Extract function info from a groundhog HPC Python file.

    Looks for functions/methods decorated with @hog.function or @hog.method.
    For methods, returns the fully qualified name as ClassName.method_name.
    Falls back to filename if no decorated function is found.
    """
    content = file_path.read_text()
    tree = ast.parse(content)

    # First, look for @hog.function decorated functions at module level
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and _has_hog_decorator(node):
            return {
                "function_name": node.name,
                "docstring": ast.get_docstring(node) or "",
                "function_text": content,
            }

    # Then, look for @hog.method decorated methods inside classes
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            for class_node in ast.iter_child_nodes(node):
                if isinstance(class_node, ast.FunctionDef) and _has_hog_decorator(
                    class_node
                ):
                    return {
                        "function_name": f"{class_name}.{class_node.name}",
                        "docstring": ast.get_docstring(class_node) or "",
                        "function_text": content,
                    }

    # Fallback to filename
    return {
        "function_name": file_path.stem,
        "docstring": "",
        "function_text": content,
    }


# =============================================================================
# Endpoint Commands
# =============================================================================


@endpoint_app.command("create")
def create_endpoint(
    name: str = typer.Option(..., "--name", "-n", help="Endpoint name"),
    gcmu_id: Optional[str] = typer.Option(
        None, "--gcmu-id", "-g", help="Globus Compute endpoint UUID"
    ),
):
    """Register a new HPC endpoint."""
    client = GardenClient()

    request = HpcEndpointCreateRequest(name=name, gcmu_id=gcmu_id)
    endpoint = client.backend_client.create_hpc_endpoint(request)

    rich.print("[green]✓[/green] Endpoint created successfully!")
    rich.print(f"  ID: [bold]{endpoint.id}[/bold]")
    rich.print(f"  Name: {endpoint.name}")
    if endpoint.gcmu_id:
        rich.print(f"  GCMU ID: {endpoint.gcmu_id}")


@endpoint_app.command("list")
def list_endpoints(
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum results"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON output"),
):
    """List available HPC endpoints."""
    client = GardenClient()
    endpoints = client.backend_client.get_hpc_endpoints(limit=limit)

    if json_output:
        data = [ep.model_dump(mode="json") for ep in endpoints]
        if pretty:
            rich.print(data)
        else:
            print(json.dumps(data))
        return

    if not endpoints:
        rich.print("[yellow]No endpoints found.[/yellow]")
        return

    table = Table(title="HPC Endpoints")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("GCMU ID")
    table.add_column("Owner")

    for ep in endpoints:
        table.add_row(
            str(ep.id),
            ep.name,
            ep.gcmu_id or "-",
            ep.owner or "-",
        )

    rich.print(table)


@endpoint_app.command("show")
def show_endpoint(
    endpoint_id: int = typer.Argument(..., help="Endpoint ID"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON output"),
):
    """Show details of a HPC endpoint."""
    client = GardenClient()
    endpoint = client.backend_client.get_hpc_endpoint(endpoint_id)

    if json_output:
        data = endpoint.model_dump(mode="json")
        if pretty:
            rich.print(data)
        else:
            print(json.dumps(data))
        return

    rich.print(f"\n[bold]Endpoint: {endpoint.name}[/bold]")
    rich.print(f"  ID: [cyan]{endpoint.id}[/cyan]")
    if endpoint.gcmu_id:
        rich.print(f"  GCMU ID: {endpoint.gcmu_id}")
    if endpoint.owner:
        rich.print(f"  Owner: {endpoint.owner}")


@endpoint_app.command("update")
def update_endpoint(
    endpoint_id: int = typer.Argument(..., help="Endpoint ID"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="New name"),
    gcmu_id: Optional[str] = typer.Option(None, "--gcmu-id", "-g", help="New GCMU ID"),
):
    """Update a HPC endpoint."""
    client = GardenClient()

    request = HpcEndpointPatchRequest(name=name, gcmu_id=gcmu_id)

    if not any([request.name, request.gcmu_id]):
        rich.print("[yellow]No updates specified.[/yellow]")
        raise typer.Exit(1)

    endpoint = client.backend_client.patch_hpc_endpoint(endpoint_id, request)

    rich.print("[green]✓[/green] Endpoint updated successfully!")
    rich.print(f"  ID: {endpoint.id}")
    rich.print(f"  Name: {endpoint.name}")


@endpoint_app.command("delete")
def delete_endpoint(
    endpoint_id: int = typer.Argument(..., help="Endpoint ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete a HPC endpoint."""
    client = GardenClient()

    endpoint = client.backend_client.get_hpc_endpoint(endpoint_id)

    if not force:
        if not typer.confirm(f"Delete endpoint '{endpoint.name}'?"):
            rich.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    client.backend_client.delete_hpc_endpoint(endpoint_id)
    rich.print(f"[green]✓[/green] Endpoint {endpoint_id} deleted successfully!")


# =============================================================================
# Function Commands
# =============================================================================


@groundhog_app.command("deploy")
def deploy_function(
    file: Path = typer.Argument(..., help="Path to Python file with function"),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Function name (auto-detected if not provided)"
    ),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="Function title"),
    endpoint_ids: str = typer.Option(
        ..., "--endpoint-ids", "-e", help="Comma-separated endpoint IDs"
    ),
    authors: Optional[str] = typer.Option(
        None, "--authors", "-a", help="Comma-separated authors"
    ),
    description: Optional[str] = typer.Option(
        None, "--description", "-d", help="Function description"
    ),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    requirements: Optional[str] = typer.Option(
        None, "--requirements", "-r", help="Comma-separated pip requirements"
    ),
):
    """Deploy a HPC function from a Python file."""
    if not file.exists():
        rich.print(f"[red]Error:[/red] File not found: {file}")
        raise typer.Exit(1)

    client = GardenClient()

    # Parse endpoint IDs
    ep_ids = parse_int_list(endpoint_ids)
    if not ep_ids:
        rich.print("[red]Error:[/red] At least one endpoint ID is required.")
        raise typer.Exit(1)

    # Extract function info from file
    fn_info = _extract_function_from_file(file)
    function_name = name or fn_info["function_name"]
    function_title = title or function_name

    year = str(datetime.now().year)
    author_list = parse_list(authors) or [client.get_email()]

    request = HpcFunctionCreateRequest(
        function_name=function_name,
        endpoint_ids=ep_ids,
        function_text=fn_info["function_text"],
        title=function_title,
        description=description or fn_info["docstring"] or None,
        year=year,
        authors=author_list,
        tags=parse_list(tags),
        requirements=parse_list(requirements),
    )

    rich.print(f"\n[bold]Deploying HPC function:[/bold] {function_name}")

    fn = client.backend_client.create_hpc_function(request)

    rich.print("\n[green]✓[/green] Function deployed successfully!")
    rich.print(f"  ID: [bold]{fn.id}[/bold]")
    rich.print(f"  Name: {fn.function_name}")
    rich.print(f"  Title: {fn.title}")
    if fn.available_endpoints:
        ep_names = [ep.name for ep in fn.available_endpoints]
        rich.print(f"  Endpoints: {', '.join(ep_names)}")


@groundhog_app.command("list")
def list_functions(
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON output"),
):
    """List your HPC functions."""
    client = GardenClient()
    functions = client.backend_client.get_hpc_functions()

    if json_output:
        data = [fn.model_dump(mode="json") for fn in functions]
        if pretty:
            rich.print(data)
        else:
            print(json.dumps(data))
        return

    if not functions:
        rich.print("[yellow]No HPC functions found.[/yellow]")
        return

    table = Table(title="HPC Functions")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Title")
    table.add_column("Endpoints")
    table.add_column("Invocations")

    for fn in functions:
        ep_count = len(fn.available_endpoints)
        table.add_row(
            str(fn.id),
            fn.function_name,
            (fn.title[:30] + "...")
            if fn.title and len(fn.title) > 30
            else (fn.title or "-"),
            str(ep_count),
            str(fn.num_invocations),
        )

    rich.print(table)


@groundhog_app.command("show")
def show_function(
    function_id: int = typer.Argument(..., help="Function ID"),
    show_code: bool = typer.Option(False, "--code", "-c", help="Show function code"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON output"),
):
    """Show details of a HPC function."""
    client = GardenClient()
    fn = client.backend_client.get_hpc_function(function_id)

    if json_output:
        data = fn.model_dump(mode="json")
        if pretty:
            rich.print(data)
        else:
            print(json.dumps(data))
        return

    rich.print(f"\n[bold]HPC Function: {fn.function_name}[/bold]")
    rich.print(f"  ID: [cyan]{fn.id}[/cyan]")
    if fn.title:
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

    if fn.available_endpoints:
        rich.print("\n  [bold]Available Endpoints:[/bold]")
        for ep in fn.available_endpoints:
            gcmu = f" (GCMU: {ep.gcmu_id})" if ep.gcmu_id else ""
            rich.print(f"    - {ep.name}{gcmu}")

    if show_code and fn.function_text:
        rich.print("\n  [bold]Function Code:[/bold]")
        from rich.syntax import Syntax

        syntax = Syntax(fn.function_text, "python", theme="monokai", line_numbers=True)
        rich.print(syntax)


@groundhog_app.command("update")
def update_function(
    function_id: int = typer.Argument(..., help="Function ID"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="New function name"),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="New title"),
    description: Optional[str] = typer.Option(
        None, "--description", "-d", help="New description"
    ),
    authors: Optional[str] = typer.Option(
        None, "--authors", "-a", help="New comma-separated authors"
    ),
    tags: Optional[str] = typer.Option(None, "--tags", help="New comma-separated tags"),
    endpoint_ids: Optional[str] = typer.Option(
        None, "--endpoint-ids", "-e", help="New comma-separated endpoint IDs"
    ),
):
    """Update a HPC function's metadata."""
    client = GardenClient()

    request = HpcFunctionPatchRequest(
        function_name=name,
        title=title,
        description=description,
        authors=parse_list(authors) if authors else None,
        tags=parse_list(tags) if tags else None,
        endpoint_ids=parse_int_list(endpoint_ids) if endpoint_ids else None,
    )

    if not any(
        [
            request.function_name,
            request.title,
            request.description,
            request.authors,
            request.tags,
            request.endpoint_ids,
        ]
    ):
        rich.print("[yellow]No updates specified.[/yellow]")
        raise typer.Exit(1)

    fn = client.backend_client.patch_hpc_function(function_id, request)

    rich.print("[green]✓[/green] Function updated successfully!")
    rich.print(f"  ID: {fn.id}")
    rich.print(f"  Name: {fn.function_name}")


@groundhog_app.command("delete")
def delete_function(
    function_id: int = typer.Argument(..., help="Function ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete a HPC function."""
    client = GardenClient()

    fn = client.backend_client.get_hpc_function(function_id)

    if not force:
        if not typer.confirm(f"Delete function '{fn.function_name}'?"):
            rich.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    client.backend_client.delete_hpc_function(function_id)
    rich.print(f"[green]✓[/green] Function {function_id} deleted successfully!")
