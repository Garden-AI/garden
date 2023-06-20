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
    GARDEN_INDEX_UUID,
    RemoteGardenException,
)
from garden_ai.gardens import Garden
from garden_ai.pipelines import RegisteredPipeline
from garden_ai.app.console import console, get_local_garden_rich_table

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
        None,
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

    local_data.put_local_garden(garden)

    if verbose:
        metadata = json.dumps(local_data.get_local_garden_by_doi(garden.doi))
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
        logger.fatal(f"Could not query search index {GARDEN_INDEX_UUID}")
        logger.fatal(e.error_data)
        raise typer.Exit(code=1) from e

    rich.print_json(results)


@garden_app.command(no_args_is_help=True)
def add_pipeline(
    garden_id: str = typer.Option(
        ...,
        "-g",
        "--garden",
        prompt="Please enter the DOI of a garden",
        help="The name of the garden you want to add a pipeline to",
        rich_help_panel="Required",
    ),
    pipeline_id: str = typer.Option(
        ...,
        "-p",
        "--pipeline",
        prompt="Please enter the DOI of a pipeline",
        help="The name of the pipeline you want to add",
        rich_help_panel="Required",
    ),
    pipeline_alias: Optional[str] = typer.Option(
        None,
        "-a",
        "--alias",
        help=(
            'Alternate short_name to use when calling this pipeline as a "method" of the'
            "garden, e.g. ``my_garden.alias(args, endpoint=...)``. Defaults to the variable"
            "name used when the pipeline was first registered."
        ),
    ),
):
    """Add a registered pipeline to a garden"""

    garden = _get_garden(garden_id)
    if not garden:
        raise typer.Exit(code=1)
    to_add = _get_pipeline(pipeline_id)
    if not to_add:
        raise typer.Exit(code=1)

    if to_add in garden.pipelines:
        if pipeline_alias:
            old_name = (
                garden.pipeline_aliases.get(to_add.short_name) or to_add.short_name
            )
            print(
                f"Pipeline {pipeline_id} is already in Garden {garden_id} as {old_name}. Renaming to {pipeline_alias}."
            )
            garden.rename_pipeline(old_name, pipeline_alias)
    else:
        garden.pipeline_ids += [to_add.doi]
        if pipeline_alias:
            garden.rename_pipeline(to_add.short_name, pipeline_alias)
    local_data.put_local_garden(garden)
    logger.info(f"Added pipeline {pipeline_id} to Garden {garden_id}")


@garden_app.command(no_args_is_help=True)
def publish(
    garden_id: str = typer.Option(
        ...,
        "-g",
        "--garden",
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


@garden_app.command(no_args_is_help=False)
def list():
    """Lists all local Gardens."""

    resource_table_cols = ["doi", "title", "description"]
    table_name = "Local Gardens"

    table = get_local_garden_rich_table(
        resource_table_cols=resource_table_cols, table_name=table_name
    )
    console.print("\n")
    console.print(table)


def _get_pipeline(pipeline_id: str) -> Optional[RegisteredPipeline]:
    pipeline = local_data.get_local_pipeline_by_doi(pipeline_id)
    if not pipeline:
        logger.warning(f"Could not find pipeline with id {pipeline_id}")
        return None
    return pipeline


@garden_app.command(no_args_is_help=True)
def show(
    garden_ids: List[str] = typer.Argument(
        ...,
        help="The DOIs of the Gardens you want to show the local data for. "
        "e.g. ``garden show garden1_doi garden2_doi`` will show the local data for both Gardens listed.",
    ),
):
    """Shows all info for some Gardens"""

    for garden_id in garden_ids:
        garden = _get_garden(garden_id)
        if garden:
            rich.print(f"Garden: {garden_id} local data:")
            rich.print_json(json=garden.json())
            rich.print("\n")


def _get_garden(garden_id: str) -> Optional[Garden]:
    garden = local_data.get_local_garden_by_doi(garden_id)
    if not garden:
        logger.warning(f"Could not find garden with id {garden_id}")
        return None
    return garden


def create_query(
    title: Optional[str] = None,
    authors: List[str] = None,
    year: str = None,
    contributors: List[str] = None,
    description: Optional[str] = None,
    tags: List[str] = None,
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
