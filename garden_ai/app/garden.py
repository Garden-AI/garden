import json
import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import rich
import typer
from globus_sdk import SearchAPIError
from rich.prompt import Prompt

from garden_ai import local_data
from garden_ai.client import GardenClient
from garden_ai.gardens import Garden
from garden_ai.pipelines import RegisteredPipeline

logger = logging.getLogger()

garden_app = typer.Typer(name="garden", no_args_is_help=True)


def setup_directory(directory: Optional[Path]) -> Optional[Path]:
    """
    Validate the directory provided by the user, scaffolding with "pipelines/" and
    "models/" subdirectories if possible (i.e. directory does not yet exist or
    exists but is empty).
    """
    if directory is None:
        return None

    if directory.exists() and any(directory.iterdir()):
        logger.fatal("Directory must be empty if it already exists.")
        raise typer.Exit(code=1)

    (directory / "models").mkdir(parents=True)
    (directory / "pipelines").mkdir(parents=True)

    with open(directory / "models" / ".gitignore", "w") as f_out:
        f_out.write("# TODO\n")

    with open(directory / "README.md", "w") as f_out:
        f_out.write("# TODO\n")

    return directory


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
    directory: Path = typer.Argument(
        None,
        callback=setup_directory,
        dir_okay=True,
        file_okay=False,
        writable=True,
        readable=True,
        resolve_path=True,
        help=(
            "(Optional) if specified, this generates a directory with subfolders to help organize the new Garden. "
            "This is likely to be useful if you want to track your Garden/Pipeline development with GitHub."
        ),
    ),
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
        metadata = json.dumps(local_data.get_local_garden_by_uuid(garden.uuid))
        rich.print_json(metadata)
    return


@garden_app.command(no_args_is_help=True)
def add_pipeline(
    garden_id: str = typer.Option(
        ...,
        "-g",
        "--garden",
        prompt="Please enter the UUID or DOI of a garden",
        help="The name of the garden you want to add a pipeline to",
        rich_help_panel="Required",
    ),
    pipeline_id: str = typer.Option(
        ...,
        "-p",
        "--pipeline",
        prompt="Please enter a the UUID or DOI of a pipeline",
        help="The name of the pipeline you want to add",
        rich_help_panel="Required",
    ),
):
    """ "Add a registered pipeline to a garden"""

    garden = get_garden(garden_id)
    to_add = get_pipeline(pipeline_id)

    if to_add in garden.pipelines:
        logger.info(f"Pipeline {pipeline_id} is already in Garden {garden_id}")
        return

    garden.pipelines += [to_add]
    local_data.put_local_garden(garden)
    logger.info(f"Added pipeline {pipeline_id} to Garden {garden_id}")


@garden_app.command(no_args_is_help=True)
def publish(
    garden_id: str = typer.Option(
        ...,
        "-g",
        "--garden",
        prompt="Please enter the UUID or DOI of a garden",
        help="The name of the garden you want to add a pipeline to",
        rich_help_panel="Required",
    ),
):
    """Push data about a Garden stored to Globus Search so that other users can search for it"""

    client = GardenClient()
    garden = get_garden(garden_id)
    garden.doi = client._mint_doi(garden)

    try:
        client.publish_garden_metadata(garden.dict())
    except SearchAPIError as e:
        logger.fatal(f"Could not publish garden {garden_id}")
        logger.fatal(e.error_data)
        raise typer.Exit(code=1) from e


def get_pipeline(pipeline_id: str) -> RegisteredPipeline:
    if "/" in pipeline_id:
        pipeline = local_data.get_local_pipeline_by_doi(pipeline_id)
    else:
        pipeline = local_data.get_local_pipeline_by_uuid(pipeline_id)
    if not pipeline:
        logger.fatal(f"Could not find pipeline with id {pipeline_id}")
        raise typer.Exit(code=1)
    return pipeline


def get_garden(garden_id: str) -> Garden:
    if "/" in garden_id:
        garden = local_data.get_local_garden_by_doi(garden_id)
    else:
        garden = local_data.get_local_garden_by_uuid(garden_id)
    if not garden:
        logger.fatal(f"Could not find garden with id {garden_id}")
        raise typer.Exit(code=1)
    return garden
