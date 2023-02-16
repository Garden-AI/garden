import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import rich
import typer
from garden_ai.client import GardenClient
from rich.prompt import Prompt

logger = logging.getLogger()

LOCAL_STORAGE = Path("~/.garden/db/").expanduser()
(LOCAL_STORAGE / "gardens").mkdir(parents=True, exist_ok=True)
(LOCAL_STORAGE / "pipelines").mkdir(parents=True, exist_ok=True)


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


# @app.callback()
def create_garden(
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
    title: str = typer.Option(
        ...,
        "-t",
        "--title",
        prompt="Please enter a title for your Garden",
        help="Provide an official title (as it should appear in citations)",
        rich_help_panel="Required",
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

    client.register_metadata(garden, out_dir=LOCAL_STORAGE / "gardens")

    if verbose:
        with open(LOCAL_STORAGE / "gardens" / f"{garden.garden_id}.json", "r") as f_in:
            metadata = f_in.read()
            rich.print_json(metadata)
    return
