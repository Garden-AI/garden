import logging
from datetime import datetime
from typing import List, Optional

import rich
import typer
from globus_sdk import SearchAPIError
from rich.prompt import Prompt

from garden_ai.app.completion import complete_entrypoint, complete_garden
from garden_ai.app.console import console, get_owned_gardens_rich_table
from garden_ai.client import GardenClient
from garden_ai.constants import GardenConstants
from garden_ai.gardens import Garden
from garden_ai.schemas.garden import GardenMetadata
from garden_ai.utils.interactive_cli import gui_edit_garden_entity

logger = logging.getLogger()

garden_app = typer.Typer(name="garden", no_args_is_help=True)


def validate_name(name: str) -> str:
    """(this may eventually use some 3rd party name parsing library)"""
    return name.strip() if name else ""


@garden_app.callback()
def garden():
    """
    sub-commands for creating and manipulating Gardens
    """
    pass


@garden_app.command(no_args_is_help=True)
def create(
    title: str = typer.Option(
        ...,
        "-t",
        "--title",
        prompt="Please enter a title for your Garden",
        help="Provide an official title (as it should appear in citations)",
        rich_help_panel="Required",
    ),
    authors: List[str] = typer.Option(
        None,
        "-a",
        "--author",
        help=(
            "Name an author of this Garden. Repeat this to indicate multiple authors: "
            "`garden create ... --author='Mendel, Gregor' --author 'Other-Author, Anne' ...` (order is preserved)."
        ),
        rich_help_panel="Required",
        prompt=False,  # NOTE: automatic prompting won't play nice with list values
    ),
    year: str = typer.Option(
        str(datetime.now().year),  # default to current year
        "-y",
        "--year",
        rich_help_panel="Required",
    ),
    contributors: List[str] = typer.Option(
        None,
        "-c",
        "--contributor",
        help=(
            "Acknowledge a contributor in this Garden. Repeat to indicate multiple (like --author). "
        ),
        rich_help_panel="Recommended",
    ),
    description: Optional[str] = typer.Option(
        None,
        "-d",
        "--description",
        help=(
            "A brief summary of the Garden and/or its purpose, to aid discovery by other Gardeners."
        ),
        rich_help_panel="Recommended",
    ),
    tags: List[str] = typer.Option(
        [],
        "--tag",
        help="Add a tag, keyword, key phrase or other classification pertaining to the Garden.",
        rich_help_panel="Recommended",
    ),
    verbose: bool = typer.Option(
        False, help="If true, pretty-print Garden's metadata when created."
    ),
):
    """Create a new Garden"""
    while not authors:
        # repeatedly prompt for at least one author until one is given
        name = validate_name(Prompt.ask("Please enter at least one author (required)"))
        if not name:
            continue

        authors = [name]
        # prompt for additional authors until one is *not* given
        while True:
            name = validate_name(
                Prompt.ask("Add another author? (leave blank to finish)")
            )
            if name:
                authors += [name]
            else:
                break

    if not contributors:
        name = validate_name(
            Prompt.ask("Acknowledge a contributor? (leave blank to skip)")
        )
        if name:
            contributors = [name]
            while True:
                name = validate_name(
                    Prompt.ask("Add another contributor? (leave blank to finish)")
                )
                if name:
                    authors += [name]
                else:
                    break
        else:
            contributors = []

    if not description:
        description = Prompt.ask(
            "Provide a brief description of this Garden, to aid in discovery (leave blank to skip)"
        )

    client = GardenClient()

    doi = client._mint_draft_doi()

    garden: Garden = client._create_garden(
        GardenMetadata(
            doi=doi,
            authors=authors,
            title=title,
            year=year,
            description=description,
            contributors=contributors,
            tags=tags,
        )
    )
    if verbose:
        rich.print_json(garden.metadata.model_dump_json())
    rich.print(
        f"Garden '{garden.metadata.title}' created with DOI: {garden.metadata.doi}"
    )
    return


@garden_app.command(no_args_is_help=True)
def search(
    title: Optional[str] = typer.Option(
        None, "-t", "--title", help="Title of a Garden"
    ),
    authors: Optional[List[str]] = typer.Option(
        None, "-a", "--author", help="an author of the Garden"
    ),
    year: Optional[str] = typer.Option(
        None, "-y", "--year", help="year the Garden was published"
    ),
    contributors: List[str] = typer.Option(
        None,
        "-c",
        "--contributor",
        help="a contributor to the Garden",
    ),
    description: Optional[str] = typer.Option(
        None,
        "-d",
        "--description",
        help="text in the description of the Garden you are searching for",
    ),
    tags: List[str] = typer.Option(
        None,
        "--tag",
        help="A tag of the Garden",
    ),
    verbose: bool = typer.Option(
        False, help="If true, print the query being passed to Globus Search."
    ),
    raw_query: Optional[str] = typer.Option(
        None,
        help=(
            "Form your own Globus Search query directly. It will be passed to Search in advanced mode."
            "Overrides all the other query options."
            "See https://docs.globus.org/api/search/reference/get_query for more details."
        ),
    ),
):
    """Queries the Garden search index and prints matching results. All query components are ANDed together.
    So if you say `garden-ai garden search --description "foo" --title "bar"` you will get results
    for gardens that have "foo" in their description and "bar" in their title.
    """
    client = GardenClient()
    if raw_query:
        query = raw_query
    else:
        query = create_query(
            title=title,
            authors=authors,
            year=year,
            contributors=contributors,
            description=description,
            tags=tags,
        )
    if verbose:
        logger.info(query)

    try:
        results = client.search(query)
    except SearchAPIError as e:
        logger.fatal(
            f"Could not query search index {GardenConstants.GARDEN_INDEX_UUID}"
        )
        logger.fatal(e.error_data)
        raise typer.Exit(code=1) from e

    rich.print_json(results)


@garden_app.command(no_args_is_help=True)
def add_entrypoint(
    garden_doi: str = typer.Option(
        ...,
        "-g",
        "--garden",
        shell_complete=complete_garden,
        prompt="Please enter the DOI of a garden",
        help="The name of the garden you want to add an entrypoint to",
        rich_help_panel="Required",
    ),
    entrypoint_doi: str = typer.Option(
        ...,
        "-e",
        "--entrypoint",
        shell_complete=complete_entrypoint,
        prompt="Please enter the DOI of an entrypoint",
        help="The name of the entrypoint you want to add",
        rich_help_panel="Required",
    ),
    entrypoint_alias: Optional[str] = typer.Option(
        None,
        "-a",
        "--alias",
        help=(
            'Alternate short_name to use when calling this entrypoint as a "method" of the'
            "garden, e.g. ``my_garden.alias(args, endpoint=...)``. Defaults to the variable"
            "name used when the entrypoint was first registered."
        ),
    ),
    verbose: bool = typer.Option(
        False, help="If true, print Garden's metadata when successful."
    ),
):
    """Add a registered entrypoint to a garden"""
    client = GardenClient()
    updated_garden = client.add_entrypoint_to_garden(
        entrypoint_doi, garden_doi, alias=entrypoint_alias
    )
    rich.print(f"Added entrypoint {entrypoint_doi} to Garden {garden_doi}.")
    if verbose:
        rich.print(updated_garden.metadata)


@garden_app.command(no_args_is_help=True)
def delete(
    garden_doi: str = typer.Argument(
        ...,
        shell_complete=complete_garden,
        help="The DOI of the garden you want to delete.",
        rich_help_panel="Required",
    ),
):
    """Delete a Garden from thegardens.ai"""
    client = GardenClient()
    # backend handles delete permission, so just re-raise the error if not allowed to delete
    typer.confirm(
        f"You are about to delete garden {garden_doi} from thegardens.ai. "
        "Are you sure you want to proceed?",
        abort=True,
    )
    try:
        client.backend_client.delete_garden(garden_doi)
        console.print(f"Garden {garden_doi} has been deleted from thegardens.ai.")
    except Exception as e:
        raise typer.Exit(code=1) from e


@garden_app.command(no_args_is_help=True)
def register_doi(
    doi: str = typer.Argument(
        ...,
        shell_complete=complete_garden,
        help="The draft garden DOI you want to register",
        rich_help_panel="Required",
    ),
):
    """
    Publicly register a Garden's DOI, moving it out of draft state.

    NOTE: Gardens with registered DOIs cannot be deleted.
    """
    client = GardenClient()
    client.register_garden_doi(doi)
    rich.print(f"DOI {doi} has been moved out of draft status and can now be cited.")


@garden_app.command(no_args_is_help=False)
def list():
    """Lists all owned Gardens."""
    client = GardenClient()

    resource_table_cols = ["doi", "title", "description", "doi_is_draft"]
    table_name = "My Gardens"

    table = get_owned_gardens_rich_table(
        client, resource_table_cols=resource_table_cols, table_name=table_name
    )
    console.print("\n")
    console.print(table)


@garden_app.command(no_args_is_help=True)
def show(
    garden_ids: List[str] = typer.Argument(
        ...,
        help="The DOIs of the Gardens you want to show the data for. "
        "e.g. ``garden show garden1_doi garden2_doi`` will show the data for both Gardens listed.",
        shell_complete=complete_garden,
    ),
):
    """Shows all info for some Gardens"""
    client = GardenClient()
    gardens = client.backend_client.get_gardens(dois=garden_ids)
    for garden in gardens:
        rich.print(f"Garden {garden.metadata.doi} data:")
        rich.print_json(json=garden.metadata.model_dump_json())
        rich.print("\n")


@garden_app.command()
def edit(
    doi: str = typer.Argument(
        ...,
        shell_complete=complete_garden,
        help="The DOI of the garden you want to edit",
        rich_help_panel="Required",
    )
):
    """Edit a Garden's metadata"""

    client = GardenClient()
    garden_meta = client.backend_client.get_garden_metadata(doi)
    if not garden_meta:
        raise typer.Exit(code=1)

    string_fields = ["title", "description", "year"]
    list_fields = ["authors", "contributors", "tags"]

    edited_garden_meta = gui_edit_garden_entity(garden_meta, string_fields, list_fields)
    client.backend_client.put_garden(edited_garden_meta)
    console.print(f"Updated garden {doi}.")


def create_query(
    title: Optional[str] = None,
    authors: Optional[List[str]] = None,
    year: Optional[str] = None,
    contributors: Optional[List[str]] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> str:
    query_parts = []
    if title:
        query_parts.append(f'(title: "{title}")')

    if authors:
        for author in authors:
            query_parts.append(f'(authors: "{author}")')

    if year:
        query_parts.append(f'(year: "{year}")')

    if contributors:
        for contributor in contributors:
            query_parts.append(f'(contributors: "{contributor}")')

    if description:
        query_parts.append(f'(description: "{description}")')

    if tags:
        for tag in tags:
            query_parts.append(f'(tags: "{tag}")')

    return " AND ".join(query_parts)
