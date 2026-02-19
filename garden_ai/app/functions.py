"""CLI commands for unified function management (Modal + HPC)."""

from enum import Enum
from typing import Optional

import rich
import typer
from rich.table import Table

from garden_ai import GardenClient
from garden_ai.app.groundhog import groundhog_app
from garden_ai.app.modal_cmds import modal_app

functions_app = typer.Typer(help="Manage functions (Modal and HPC)")
functions_app.add_typer(modal_app, name="modal")
functions_app.add_typer(groundhog_app, name="hpc")


class FunctionType(str, Enum):
    """Function type filter."""

    CLOUD = "cloud"
    HPC = "hpc"


@functions_app.command("list")
def list_functions(
    type_filter: Optional[FunctionType] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by function type: 'cloud' (Modal) or 'hpc' (Groundhog)",
    ),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum results per type"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON output"),
):
    """List all your functions (Modal and HPC).

    Shows a unified view of all function types with a Type column to distinguish
    between Cloud (Modal) and HPC (Groundhog) functions.
    """
    client = GardenClient()

    # Collect functions based on filter
    modal_functions = []
    hpc_functions = []

    if type_filter is None or type_filter == FunctionType.CLOUD:
        modal_functions = client.backend_client.get_modal_functions(limit=limit)

    if type_filter is None or type_filter == FunctionType.HPC:
        hpc_functions = client.backend_client.get_hpc_functions()

    if json_output:
        data = []
        for fn in modal_functions:
            item = fn.model_dump(mode="json")
            item["type"] = "cloud"
            data.append(item)
        for fn in hpc_functions:
            item = fn.model_dump(mode="json")
            item["type"] = "hpc"
            data.append(item)

        if pretty:
            rich.print(data)
        else:
            import json

            print(json.dumps(data))
        return

    if not modal_functions and not hpc_functions:
        if type_filter:
            rich.print(f"[yellow]No {type_filter.value} functions found.[/yellow]")
        else:
            rich.print("[yellow]No functions found.[/yellow]")
        return

    table = Table(title="Functions")
    table.add_column("ID", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Name")
    table.add_column("Title")
    table.add_column("Authors")
    table.add_column("Invocations", justify="right")

    # Add Modal functions
    for fn in modal_functions:
        title = fn.title or "-"
        title_display = (title[:25] + "...") if len(title) > 25 else title
        authors = ", ".join(fn.authors[:2]) if fn.authors else "-"
        if len(fn.authors) > 2:
            authors += "..."

        table.add_row(
            str(fn.id),
            "[blue]Cloud[/blue]",
            fn.function_name or "-",
            title_display,
            authors,
            str(fn.num_invocations),
        )

    # Add HPC functions
    for fn in hpc_functions:
        title = fn.title or "-"
        title_display = (title[:25] + "...") if len(title) > 25 else title
        authors = ", ".join(fn.authors[:2]) if fn.authors else "-"
        if len(fn.authors) > 2:
            authors += "..."

        table.add_row(
            str(fn.id),
            "[green]HPC[/green]",
            fn.function_name or "-",
            title_display,
            authors,
            str(fn.num_invocations),
        )

    rich.print(table)

    # Show summary
    if type_filter is None:
        rich.print(
            f"\n[dim]Total: {len(modal_functions)} Cloud, {len(hpc_functions)} HPC[/dim]"
        )


@functions_app.command("show")
def show_function(
    function_id: int = typer.Argument(..., help="Function ID"),
    function_type: FunctionType = typer.Option(
        ...,
        "--type",
        "-t",
        help="Function type: 'cloud' (Modal) or 'hpc' (Groundhog)",
    ),
    show_code: bool = typer.Option(False, "--code", "-c", help="Show function code"),
):
    """Show details of a function.

    Requires specifying the function type since IDs are not unique across types.
    """
    client = GardenClient()

    if function_type == FunctionType.CLOUD:
        fn = client.backend_client.get_modal_function(function_id)

        rich.print(f"\n[bold]Modal Function: {fn.function_name}[/bold]")
        rich.print(f"  ID: [cyan]{fn.id}[/cyan]")
        rich.print("  Type: [blue]Cloud (Modal)[/blue]")
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

            syntax = Syntax(
                fn.function_text, "python", theme="monokai", line_numbers=True
            )
            rich.print(syntax)

    else:  # HPC
        fn = client.backend_client.get_hpc_function(function_id)

        rich.print(f"\n[bold]HPC Function: {fn.function_name}[/bold]")
        rich.print(f"  ID: [cyan]{fn.id}[/cyan]")
        rich.print("  Type: [green]HPC (Groundhog)[/green]")
        if fn.title:
            rich.print(f"  Title: {fn.title}")
        if fn.description:
            rich.print(f"  Description: {fn.description}")
        if fn.authors:
            rich.print(f"  Authors: {', '.join(fn.authors)}")
        if fn.tags:
            rich.print(f"  Tags: {', '.join(fn.tags)}")
        if fn.year:
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

            syntax = Syntax(
                fn.function_text, "python", theme="monokai", line_numbers=True
            )
            rich.print(syntax)
