"""CLI commands for Garden management."""

from typing import Optional

import rich
import typer
from rich.table import Table

from garden_ai import GardenClient
from garden_ai.schemas.garden import GardenCreateRequest, GardenPatchRequest

garden_app = typer.Typer(help="Manage Gardens")


def _parse_list(value: str | None) -> list[str]:
    """Parse a comma-separated string into a list."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_list(value: str | None) -> list[int]:
    """Parse a comma-separated string of integers."""
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


@garden_app.command("create")
def create_garden(
    title: str = typer.Option(..., "--title", "-t", help="Title of the garden"),
    authors: str = typer.Option(
        ..., "--authors", "-a", help="Comma-separated list of authors"
    ),
    description: Optional[str] = typer.Option(
        None, "--description", "-d", help="Description of the garden"
    ),
    tags: Optional[str] = typer.Option(
        None, "--tags", help="Comma-separated list of tags"
    ),
    modal_function_ids: Optional[str] = typer.Option(
        None, "--modal-function-ids", "-m", help="Comma-separated Modal function IDs"
    ),
    hpc_function_ids: Optional[str] = typer.Option(
        None, "--hpc-function-ids", "-g", help="Comma-separated Groundhog function IDs"
    ),
    year: Optional[str] = typer.Option(None, "--year", help="Publication year"),
    version: str = typer.Option("0.0.1", "--version", help="Garden version"),
):
    """Create a new garden."""
    client = GardenClient()

    request = GardenCreateRequest(
        title=title,
        authors=_parse_list(authors),
        description=description,
        tags=_parse_list(tags),
        modal_function_ids=_parse_int_list(modal_function_ids),
        hpc_function_ids=_parse_int_list(hpc_function_ids),
        year=year if year else None,
        version=version,
    )

    garden = client.backend_client.create_garden(request)

    rich.print("[green]✓[/green] Garden created successfully!")
    rich.print(f"  DOI: [bold]{garden.doi}[/bold]")
    rich.print(f"  Title: {garden.title}")
    if garden.modal_function_ids:
        rich.print(f"  Modal Functions: {garden.modal_function_ids}")
    if garden.hpc_function_ids:
        rich.print(f"  Groundhog Functions: {garden.hpc_function_ids}")


@garden_app.command("list")
def list_gardens(
    all_gardens: bool = typer.Option(
        False, "--all", help="List all published gardens instead of just yours"
    ),
    tags: Optional[str] = typer.Option(
        None, "--tags", help="Filter by comma-separated tags"
    ),
    authors: Optional[str] = typer.Option(
        None, "--authors", help="Filter by comma-separated authors"
    ),
    year: Optional[str] = typer.Option(None, "--year", help="Filter by year"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum results to show"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON output"),
):
    """List gardens."""
    client = GardenClient()

    owner_uuid = str(client.get_user_identity_id())
    if all_gardens:
        owner_uuid = None

    gardens = client.backend_client.get_gardens(
        tags=_parse_list(tags),
        authors=_parse_list(authors),
        year=year,
        owner_uuid=owner_uuid,
        limit=limit,
    )

    if json_output:
        data = [g.metadata.model_dump(mode="json") for g in gardens]
        if pretty:
            rich.print(data)
        else:
            import json

            print(json.dumps(data))
        return

    if not gardens:
        rich.print("[yellow]No gardens found.[/yellow]")
        return

    table = Table(title="Gardens")
    table.add_column("DOI", style="cyan")
    table.add_column("Title")
    table.add_column("Authors")
    table.add_column("State")

    for g in gardens:
        state = g.metadata.state or (
            "draft" if g.metadata.doi_is_draft else "published"
        )
        title = g.metadata.title
        table.add_row(
            g.metadata.doi,
            title[:40] + "..." if len(title) > 40 else title,
            ", ".join(g.metadata.authors[:2])
            + ("..." if len(g.metadata.authors) > 2 else ""),
            state,
        )

    rich.print(table)


@garden_app.command("search")
def search_garden(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results to show"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON output"),
):
    """Search for gardens using full-text search."""
    client = GardenClient()

    payload = {
        "q": query,
        "limit": limit,
        "offset": 0,
        "filters": [{"field_name": "is_archived", "values": ["false"]}],
    }

    results = client.backend_client.search_gardens(payload)
    gardens = results.get("garden_meta", [])

    if json_output:
        if pretty:
            rich.print(gardens)
        else:
            import json

            print(json.dumps(gardens))
        return

    if not gardens:
        rich.print("[yellow]No gardens found.[/yellow]")
        return

    table = Table(title=f"Search Results for '{query}'")
    table.add_column("DOI", style="cyan")
    table.add_column("Title")
    table.add_column("Authors")
    table.add_column("Description")

    for g in gardens:
        title = g.get("title", "")
        title_display = title[:40] + "..." if len(title) > 40 else title
        authors = g.get("authors", [])
        authors_display = ", ".join(authors[:2]) + ("..." if len(authors) > 2 else "")
        description = g.get("description", "") or ""
        desc_display = (
            description[:50] + "..." if len(description) > 50 else description
        )

        table.add_row(
            g.get("doi", "-"),
            title_display,
            authors_display,
            desc_display,
        )

    rich.print(table)


@garden_app.command("show")
def show_garden(
    doi: str = typer.Argument(..., help="DOI of the garden to show"),
):
    """Show details of a specific garden."""
    client = GardenClient()

    garden = client.backend_client.get_garden_metadata(doi)

    rich.print(f"\n[bold]Garden: {garden.title}[/bold]")
    rich.print(f"  DOI: [cyan]{garden.doi}[/cyan]")
    rich.print(
        f"  State: {garden.state or ('draft' if garden.doi_is_draft else 'published')}"
    )
    rich.print(f"  Authors: {', '.join(garden.authors)}")
    if garden.contributors:
        rich.print(f"  Contributors: {', '.join(garden.contributors)}")
    if garden.description:
        rich.print(f"  Description: {garden.description}")
    if garden.tags:
        rich.print(f"  Tags: {', '.join(garden.tags)}")
    rich.print(f"  Year: {garden.year}")
    rich.print(f"  Version: {garden.version}")

    if garden.modal_function_ids:
        rich.print(f"\n  [bold]Modal Functions:[/bold] {garden.modal_function_ids}")
    if garden.hpc_function_ids:
        rich.print(f"  [bold]Groundhog Functions:[/bold] {garden.hpc_function_ids}")


@garden_app.command("update")
def update_garden(
    doi: str = typer.Argument(..., help="DOI of the garden to update"),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="New title"),
    authors: Optional[str] = typer.Option(
        None, "--authors", "-a", help="New comma-separated list of authors"
    ),
    description: Optional[str] = typer.Option(
        None, "--description", "-d", help="New description"
    ),
    tags: Optional[str] = typer.Option(
        None, "--tags", help="New comma-separated list of tags"
    ),
    version: Optional[str] = typer.Option(None, "--version", help="New version"),
):
    """Update a garden's metadata."""
    client = GardenClient()

    request = GardenPatchRequest(
        title=title,
        authors=_parse_list(authors) if authors else None,
        description=description,
        tags=_parse_list(tags) if tags else None,
        version=version,
    )

    # Only send if at least one field is set
    if not any(
        [
            request.title,
            request.authors,
            request.description,
            request.tags,
            request.version,
        ]
    ):
        rich.print("[yellow]No updates specified.[/yellow]")
        raise typer.Exit(1)

    garden = client.backend_client.patch_garden(doi, request)

    rich.print("[green]✓[/green] Garden updated successfully!")
    rich.print(f"  DOI: [bold]{garden.doi}[/bold]")


@garden_app.command("add-functions")
def add_functions(
    doi: str = typer.Argument(..., help="DOI of the garden"),
    modal_function_ids: Optional[str] = typer.Option(
        None,
        "--modal-function-ids",
        "-m",
        help="Comma-separated Modal function IDs to add",
    ),
    hpc_function_ids: Optional[str] = typer.Option(
        None,
        "--hpc-function-ids",
        "-g",
        help="Comma-separated Groundhog function IDs to add",
    ),
    replace: bool = typer.Option(
        False, "--replace", help="Replace existing functions instead of adding"
    ),
):
    """Add functions to an existing garden."""
    client = GardenClient()

    modal_ids = _parse_int_list(modal_function_ids)
    hpc_ids = _parse_int_list(hpc_function_ids)

    if not modal_ids and not hpc_ids:
        rich.print("[yellow]No function IDs specified.[/yellow]")
        raise typer.Exit(1)

    if not replace:
        # Get current garden to merge function IDs
        current = client.backend_client.get_garden_metadata(doi)
        if modal_ids:
            modal_ids = list(set(current.modal_function_ids + modal_ids))
        if hpc_ids:
            hpc_ids = list(set(current.hpc_function_ids + hpc_ids))

    request = GardenPatchRequest(
        modal_function_ids=modal_ids if modal_ids else None,
        hpc_function_ids=hpc_ids if hpc_ids else None,
    )

    garden = client.backend_client.patch_garden(doi, request)

    rich.print(
        f"[green]✓[/green] Functions {'replaced' if replace else 'added'} successfully!"
    )
    rich.print(f"  Modal Functions: {garden.modal_function_ids}")
    rich.print(f"  Groundhog Functions: {garden.hpc_function_ids}")


@garden_app.command("delete")
def delete_garden(
    doi: str = typer.Argument(..., help="DOI of the garden to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
):
    """Delete a garden."""
    if not force:
        confirm = typer.confirm(f"Are you sure you want to delete garden {doi}?")
        if not confirm:
            rich.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    client = GardenClient()
    client.backend_client.delete_garden(doi)

    rich.print(f"[green]✓[/green] Garden {doi} deleted successfully!")
