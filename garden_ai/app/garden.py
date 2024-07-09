import json
import logging
from datetime import datetime
from typing import List, Optional

import rich
import typer
from globus_sdk import SearchAPIError
from rich.prompt import Prompt

from garden_ai import local_data
from garden_ai.client import GardenClient
from garden_ai.globus_search.garden_search import (
    RemoteGardenException,
)
from garden_ai.utils.dois import is_doi_registered
from garden_ai.utils.interactive_cli import gui_edit_garden_entity
from garden_ai.constants import GardenConstants
from garden_ai.gardens import Garden, PublishedGarden
from garden_ai.entrypoints import RegisteredEntrypoint
from garden_ai.app.console import (
    console,
    get_local_garden_rich_table,
    DOI_STATUS_COLUMN,
)
from garden_ai.app.completion import complete_garden, complete_entrypoint

logger = logging.getLogger()

garden_app = typer.Typer(name="garden", no_args_is_help=True)


def validate_name(name: str) -> str:
    """(this will probably eventually use some 3rd party name parsing library)"""
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

    garden = client.create_garden(
        authors=authors,
        title=title,
        year=year,
        description=description,
        contributors=contributors,
        tags=tags,
    )

    _put_garden(garden)

    if verbose:
        metadata = json.dumps(_get_garden(garden.doi))
        rich.print_json(metadata)

    if garden.doi:
        rich.print(f"Garden '{garden.title}' created with DOI: {garden.doi}")

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
    garden_id: str = typer.Option(
        ...,
        "-g",
        "--garden",
        autocompletion=complete_garden,
        prompt="Please enter the DOI of a garden",
        help="The name of the garden you want to add an entrypoint to",
        rich_help_panel="Required",
    ),
    entrypoint_id: str = typer.Option(
        ...,
        "-p",
        "--entrypoint",
        autocompletion=complete_entrypoint,
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
):
    """Add a registered entrypoint to a garden"""

    garden = _get_garden(garden_id)
    if not garden:
        raise typer.Exit(code=1)
    to_add = _get_entrypoint(entrypoint_id)
    if not to_add:
        raise typer.Exit(code=1)

    if to_add.doi in garden.entrypoint_ids:
        if entrypoint_alias:
            old_name = garden.entrypoint_aliases.get(to_add.doi) or to_add.short_name
            print(
                f"Entrypoint {entrypoint_id} is already in Garden {garden_id} as {old_name}. Renaming to {entrypoint_alias}."
            )
            garden.rename_entrypoint(to_add.doi, entrypoint_alias)
    else:
        garden.add_entrypoint(entrypoint_id, entrypoint_alias)
    _put_garden(garden)
    logger.info(f"Added entrypoint {entrypoint_id} to Garden {garden_id}")


@garden_app.command(no_args_is_help=True)
def publish(
    garden_id: str = typer.Option(
        ...,
        "-g",
        "--garden",
        autocompletion=complete_garden,
        prompt="Please enter the DOI of a garden",
        help="The DOI of the garden you want to publish",
        rich_help_panel="Required",
    ),
):
    """Push data about a Garden stored to Globus Search so that other users can search for it"""

    client = GardenClient()
    garden = _get_garden(garden_id)
    if not garden:
        raise typer.Exit(code=1)
    try:
        client.publish_garden_metadata(garden)
    except RemoteGardenException as e:
        logger.fatal(f"Could not publish garden {garden_id}")
        logger.fatal(str(e))
        raise typer.Exit(code=1) from e
    console.print(
        f"Successfully published garden {garden.title} with DOI {garden.doi}!"
    )


@garden_app.command(no_args_is_help=True)
def delete(
    garden_id: str = typer.Option(
        ...,
        "-g",
        "--garden",
        autocompletion=complete_garden,
        prompt="Please enter the DOI of a garden",
        help="The DOI of the garden you want to publish",
        rich_help_panel="Required",
    ),
    dangerous_override: bool = typer.Option(
        False,
        "--dangerous-override",
        help=(
            "Power users only! Deletes a garden even if it is not in your local data.json file."
        ),
        hidden=True,
    ),
):
    """Delete a Garden from your local storage and the thegardens.ai website"""
    client = GardenClient()
    if local_data._IS_DISABLED:
        # revamped api handles delete permission, so just re-raise the error if not allowed to delete
        typer.confirm(
            f"You are about to delete garden {garden_id} from the thegardens.ai search index. "
            "Are you sure you want to proceed?",
            abort=True,
        )
        try:
            client.backend_client.delete_garden(garden_id)
            client.delete_garden_from_search_index(garden_id)
        except Exception as e:
            raise typer.Exit(code=1) from e
        console.print(f"Garden {garden_id} has been deleted from thegardens.ai.")
        return

    # Does the user have this garden locally?
    garden = _get_garden(garden_id)

    # No, and they're not using the override
    if not garden and not dangerous_override:
        raise typer.Exit(code=1)

    # Is the garden DOI not in a draft state?
    is_registered = is_doi_registered(garden_id)
    if is_registered:
        if garden:
            # It has a registered DOI and they have the garden locally.
            typer.confirm(
                f"The DOI {garden_id} is registered, so we can't remove it from the search index. "
                "You can still delete it from your local data. "
                "Would you like to delete it locally?",
                abort=True,
            )
            client.delete_garden_locally(garden_id)
            console.print(f"Garden {garden_id} has been deleted locally.")
            return

        if not garden:
            # It has a registered DOI and they don't have the garden locally - nothing to do
            console.print(
                f"The DOI {garden_id} is registered, so we can't remove it from the search index. "
                "You also don't have it in your local data. "
                "There is nothing to delete.",
            )
            raise typer.Exit(code=1)

    if garden:
        # They have the garden locally and it was just a draft DOI.
        # We can delete from both places.
        typer.confirm(
            f"You are about to delete garden {garden_id} ({garden.title}) "
            "from your local data and the thegardens.ai search index.\n"
            f"Are you sure you want to proceed?",
            abort=True,
        )
        client.delete_garden_locally(garden_id)
        client.delete_garden_from_search_index(garden_id)
        console.print(
            f"Garden {garden_id} has been deleted locally and from thegardens.ai."
        )
        return

    if not garden and dangerous_override:
        # They do not have the garden locally, but they have used the override.
        # It's a draft DOI so we can remove the record from our search index.
        typer.confirm(
            f"You are about to delete garden {garden_id} from the thegardens.ai search index. "
            "Are you sure you want to proceed?",
            abort=True,
        )
        client.delete_garden_from_search_index(garden_id)
        console.print(f"Garden {garden_id} has been deleted from thegardens.ai.")
        return


@garden_app.command(no_args_is_help=True)
def register_doi(
    doi: str = typer.Argument(
        ...,
        autocompletion=complete_garden,
        help="The draft garden DOI you want to register",
        rich_help_panel="Required",
    ),
):
    """
    Moves a Garden's DOI out of draft state.

    Parameters
    ----------
    doi : str
        The DOI of the garden to be registered.
    """
    client = GardenClient()
    garden = _get_garden(doi)
    if not garden:
        raise typer.Exit(code=1)
    garden.doi_is_draft = False
    client.publish_garden_metadata(garden, register_doi=True)
    _put_garden(garden)
    rich.print(f"DOI {doi} has been moved out of draft status and can now be cited.")


if not local_data._IS_DISABLED:
    # this subcommand is no longer meaningful when local_data is disabled.
    # we can replace this with "list my gardens" behavior once
    # https://github.com/Garden-AI/garden-backend/issues/111 is live.
    @garden_app.command(no_args_is_help=False)
    def list():
        """Lists all local Gardens."""

        resource_table_cols = ["doi", "title", "description", DOI_STATUS_COLUMN]
        table_name = "Local Gardens"

        table = get_local_garden_rich_table(
            resource_table_cols=resource_table_cols, table_name=table_name
        )
        console.print("\n")
        console.print(table)


def _get_entrypoint(entrypoint_id: str) -> Optional[RegisteredEntrypoint]:
    if local_data._IS_DISABLED:
        client = GardenClient()
        entrypoint = client.backend_client.get_entrypoint(entrypoint_id)
    else:
        entrypoint = local_data.get_local_entrypoint_by_doi(entrypoint_id)  # type: ignore[assignment]
    if not entrypoint:
        logger.warning(f"Could not find entrypoint with id {entrypoint_id}")
        return None
    return entrypoint


@garden_app.command(no_args_is_help=True)
def show(
    garden_ids: List[str] = typer.Argument(
        ...,
        help="The DOIs of the Gardens you want to show the local data for. "
        "e.g. ``garden show garden1_doi garden2_doi`` will show the local data for both Gardens listed.",
        autocompletion=complete_garden,
    ),
):
    """Shows all info for some Gardens"""

    for garden_id in garden_ids:
        garden = _get_garden(garden_id)
        if garden:
            rich.print(f"Garden: {garden_id} local data:")
            rich.print_json(json=garden.model_dump_json())
            rich.print("\n")


@garden_app.command()
def edit(
    doi: str = typer.Argument(
        ...,
        autocompletion=complete_garden,
        help="The DOI of the garden you want to edit",
        rich_help_panel="Required",
    )
):
    """Edit a Garden's metadata"""

    garden = _get_garden(doi)
    if not garden:
        raise typer.Exit(code=1)

    string_fields = ["title", "description", "year"]
    list_fields = ["authors", "contributors", "tags"]

    edited_garden = gui_edit_garden_entity(garden, string_fields, list_fields)

    _put_garden(edited_garden)
    console.print(
        f"Updated garden {doi}. Run `garden-ai garden publish -g {doi}` to push the change to thegardens.ai"
    )


def _get_garden(garden_id: str) -> Optional[Garden]:
    if local_data._IS_DISABLED:
        client = GardenClient()
        published: PublishedGarden = client.backend_client.get_garden(garden_id)
        # keep contract consistent with local_data equivalent, which returns a plain Garden
        # note: the _entrypoints and kwarg is to skip the redundant/expensive
        # _collect_entrypoints call in Garden.__init__
        garden = Garden(
            **published.model_dump(),
            entrypoint_ids=[ep.doi for ep in published.entrypoints],
            _entrypoints=published.entrypoints,
        )
    else:
        garden = local_data.get_local_garden_by_doi(garden_id)  # type: ignore[assignment]
    if not garden:
        logger.warning(f"Could not find local garden with id {garden_id}")
        return None
    return garden


def _put_garden(garden):
    if local_data._IS_DISABLED:
        client = GardenClient()
        client.backend_client.update_garden(garden)
    else:
        local_data.put_local_garden(garden)


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
