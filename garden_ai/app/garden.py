import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from copy import deepcopy

import rich
import typer
from garden_ai.client import GardenClient
from garden_ai import local_data
from garden_ai.gardens import Garden
from rich.prompt import Prompt

from globus_sdk import SearchAPIError

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
        metadata = local_data.get_local_garden(garden.uuid)
        rich.print_json(metadata)
    return


# TODO: allow selecting by DOI.
@garden_app.command(no_args_is_help=True)
def add_pipeline(
    garden_uuid: str = typer.Option(
        ...,
        "-g",
        "--garden",
        prompt="Please enter a the name or DOI of a garden",
        help="The name of the garden you want to add a pipeline to",
        rich_help_panel="Required",
    ),
    pipeline_uuid: str = typer.Option(
        ...,
        "-p",
        "--pipeline",
        prompt="Please enter a the title or DOI of a pipeline",
        help="The name of the pipeline you want to add",
        rich_help_panel="Required",
    ),
):
    garden_metadata = get_garden_meta(garden_uuid)
    pipeline_uuids_already_present = set(
        pipeline["uuid"] for pipeline in garden_metadata["pipelines"]
    )
    if pipeline_uuid in pipeline_uuids_already_present:
        logger.info(f"Pipeline {pipeline_uuid} is already in Garden {garden_uuid}")
        return

    pipeline_meta = get_pipeline_meta(pipeline_uuid)
    garden_metadata["pipelines"].append(
        {"uuid": pipeline_uuid, "doi": pipeline_meta["doi"]}
    )
    local_data.put_local_garden_metadata(garden_metadata)
    logger.info(f"Added pipeline {pipeline_uuid} to Garden {garden_uuid}")


@garden_app.command(no_args_is_help=True)
def publish(
    garden_uuid: str = typer.Option(
        ...,
        "-g",
        "--garden",
        prompt="Please enter a the name or DOI of a garden",
        help="The name of the garden you want to add a pipeline to",
        rich_help_panel="Required",
    ),
):
    # TODO: mint DOI for the Garden in this command.
    client = GardenClient()

    garden_metadata = get_garden_meta(garden_uuid)
    pipeline_metas = [
        get_pipeline_meta(p["uuid"]) for p in garden_metadata["pipelines"]
    ]
    garden_metadata["pipelines"] = pipeline_metas
    garden_metadata["doi"] = mint_doi_from_garden_meta(garden_metadata, client)

    try:
        client.publish_garden_metadata(garden_metadata)
    except SearchAPIError as e:
        logger.fatal(f"Could not publish garden {garden_uuid}")
        logger.fatal(e.error_data)
        raise typer.Exit(code=1) from e


# Right now we can make a Garden model from Garde JSON iff it has no pipelines.
# Make a pipeline-less copy of the JSON so that we can make a Garden model
# and use its DOI generating method.
# TODO: clean up this hack once we fix Pydantic model deserialization.
def mint_doi_from_garden_meta(garden_meta: Dict, client: GardenClient):
    garden_copy = deepcopy(garden_meta)
    garden_copy["pipelines"] = []
    # No DOI currently serializes to None, which is not accepted by the Garden model.
    if not garden_copy["doi"]:
        garden_copy["doi"] = ""
    garden_model = Garden(**garden_copy)
    doi = client._mint_doi(garden_model)
    return doi


def get_pipeline_meta(pipeline_uuid: str):
    maybe_pipeline = local_data.get_local_pipeline(pipeline_uuid)
    if not maybe_pipeline:
        logger.fatal(f"Could not find garden with uuid {pipeline_uuid}")
        raise typer.Exit(code=1)
    try:
        pipeline_metadata = json.loads(str(maybe_pipeline))
    except json.JSONDecodeError as e:
        logger.fatal(
            f"Malformed local database. Could not parse record for {pipeline_uuid}"
        )
        raise typer.Exit(code=1) from e
    return pipeline_metadata


def get_garden_meta(garden_uuid: str) -> Dict:
    maybe_garden = local_data.get_local_garden(garden_uuid)
    if not maybe_garden:
        logger.fatal(f"Could not find garden with uuid {garden_uuid}")
        raise typer.Exit(code=1)
    try:
        garden_metadata = json.loads(str(maybe_garden))
    except json.JSONDecodeError as e:
        logger.fatal(
            f"Malformed local database. Could not parse record for {garden_uuid}"
        )
        raise typer.Exit(code=1) from e
    return garden_metadata
