import logging
import re
from datetime import datetime
from keyword import iskeyword
from pathlib import Path
from typing import List, Optional

import jinja2
import rich
import typer
from garden_ai import GardenClient, Pipeline, step
from rich import print
from rich.prompt import Prompt

logger = logging.getLogger()

pipeline_app = typer.Typer(name="pipeline", no_args_is_help=True)


def clean_identifier(name: str) -> str:
    """Clean the name provided for use as a pipeline's python identifier."""
    orig = name
    # Remove invalid characters, replacing with _
    name = re.sub("[^0-9a-zA-Z_]", "_", name)

    # Remove leading characters until we find a letter
    name = re.sub("^[^a-zA-Z]+", "", name)

    # Remove doubled/trailing underscores
    name = re.sub("__+", "_", name).rstrip("_")

    if not name:
        # name consisted only of invalid characters
        raise typer.BadParameter(
            "Invalid shortname. This argument should contain a valid python identifier"
            "(i.e. something usable as a variable name)."
        )

    # truncate to sane length, though not strictly necessary
    name = name[:50]

    if iskeyword(name):
        name += "_"

    if name != orig:
        print(f'Generated valid shortname "{name}" from "{orig}".')

    return name


def parse_full_name(name: str) -> str:
    """(this will probably eventually use some 3rd party name parsing library)"""
    return name.strip() if name else ""


def template_pipeline(shortname: str, pipeline: Pipeline) -> str:
    """populate jinja2 template with starter code for creating a pipeline"""
    env = jinja2.Environment(loader=jinja2.PackageLoader("garden_ai"))
    template = env.get_template("pipeline")
    return template.render(shortname=shortname, pipeline=pipeline)


@pipeline_app.callback()
def pipeline():
    """
    sub-commands for creating and manipulating Pipelines
    """
    pass


@pipeline_app.command(no_args_is_help=True)
def create(
    shortname: str = typer.Argument(
        None,
        help=(
            "A valid python identifier (i.e. variable name) for the new pipeline. "
            "If not provided, one will be generated from your pipeline's title. "
            "a [shortname].py file will be templated for your pipeline code. "
        ),
    ),
    directory: Path = typer.Option(
        Path.cwd(),
        dir_okay=True,
        file_okay=False,
        writable=True,
        readable=True,
        resolve_path=True,
        help=(
            "(Optional) if specified, target directory in which to generate the templated [shortname].py file. "
            "Defaults to current directory."
        ),
    ),
    title: str = typer.Option(
        ...,
        "-t",
        "--title",
        help=("Provide an official title (as it should appear in citations). "),
        rich_help_panel="Required",
        prompt="Please enter an official title for your Pipeline (as it should appear in citations)",
    ),
    authors: List[str] = typer.Option(
        None,
        "-a",
        "--author",
        help=(
            "Name an author of this Pipeline. At least one author is required. Repeat this to indicate multiple: "
            "`garden-ai pipeline create ... --author='Mendel, Gregor' --author 'Other, Anne' ...` (order is preserved)."
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
    description: Optional[str] = typer.Option(
        None,
        "--description",
        help=("A brief summary of the Pipeline and/or its purpose, to aid discovery."),
        rich_help_panel="Recommended",
    ),
    contributors: List[str] = typer.Option(
        None,
        "-c",
        "--contributor",
        help=(
            "Acknowledge a contributor to this Pipeline. Repeat to indicate multiple (like --author). "
        ),
        rich_help_panel="Recommended",
    ),
    tags: List[str] = typer.Option(
        None,
        "--tag",
        help=(
            "Add a tag, keyword, key phrase or other classification pertaining to the Pipeline. "
            "Repeat to indicate multiple (like --author). "
        ),
        rich_help_panel="Recommended",
    ),
    verbose: bool = typer.Option(
        False, help="If true, pretty-print Pipeline's metadata when created."
    ),
):
    """Scaffold a new pipeline"""

    shortname = clean_identifier(shortname or title)

    while not authors:
        # repeatedly prompt for at least one author until one is given
        name = parse_full_name(
            Prompt.ask("Please enter at least one author (required)")
        )
        if not name:
            continue

        authors = [name]
        # prompt for additional authors until one is *not* given
        while True:
            name = parse_full_name(
                Prompt.ask("Add another author? (leave blank to finish)")
            )
            if name:
                authors += [name]
            else:
                break

    if not description:
        description = Prompt.ask(
            "Provide a brief description of this Pipeline, to aid in discovery (leave blank to skip)"
        )

    client = GardenClient()

    @step
    def dummy_step(arg: object) -> object:
        """placeholder"""
        return arg

    pipeline = client.create_pipeline(
        title=title,
        authors=authors,
        contributors=contributors,
        steps=[dummy_step],  # type: ignore
        tags=tags,
        description=description,
        year=year,
    )

    # template file
    out_dir = Path(directory / shortname)
    if out_dir.exists():
        logger.fatal(f"Error: {directory / shortname} already exists.")
        raise typer.Exit(code=1)
    else:
        out_dir.mkdir(parents=True)
        out_file = out_dir / "pipeline.py"
        contents = template_pipeline(shortname, pipeline)
        with open(out_file, "w") as f:
            f.write(contents)
        print(f"Wrote to {out_file}.")

    if verbose:
        client.put_local_pipeline(pipeline)
        metadata = client.get_local_pipeline(pipeline.uuid)
        rich.print_json(metadata)

    return
