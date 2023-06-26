import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import jinja2
import typer
import rich
from rich import print
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn

from garden_ai import GardenClient, Pipeline, step, GardenConstants
from garden_ai.app.console import console, get_local_pipeline_rich_table
from garden_ai.app.garden import _get_pipeline

from garden_ai.mlmodel import PipelineLoadScaffoldedException
from garden_ai.utils.filesystem import (
    load_pipeline_from_python_file,
    PipelineLoadException,
)

from garden_ai.utils.misc import clean_identifier


logger = logging.getLogger()

pipeline_app = typer.Typer(name="pipeline", no_args_is_help=True)


def parse_full_name(name: str) -> str:
    """(this will probably eventually use some 3rd party name parsing library)"""
    return name.strip() if name else ""


def template_pipeline(short_name: str, pipeline: Pipeline) -> str:
    """populate jinja2 template with starter code for creating a pipeline"""
    env = jinja2.Environment(loader=jinja2.PackageLoader("garden_ai"))
    template = env.get_template("pipeline")
    return template.render(
        short_name=short_name,
        pipeline=pipeline,
        scaffolded_model_name=GardenConstants.SCAFFOLDED_MODEL_NAME,
    )


@pipeline_app.callback()
def pipeline():
    """
    sub-commands for creating and manipulating Pipelines
    """
    pass


@pipeline_app.command(no_args_is_help=True)
def create(
    short_name: str = typer.Argument(
        None,
        help=(
            "A valid python identifier (i.e. variable name) for the new pipeline. "
            "If not provided, one will be generated from your pipeline's title. "
            "a [short_name].py file will be templated for your pipeline code. "
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
            "(Optional) if specified, target directory in which to generate the templated [short_name].py file. "
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

    short_name = clean_identifier(short_name or title)

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
    out_dir = Path(directory / short_name)
    if out_dir.exists():
        logger.fatal(f"Error: {directory / short_name} already exists.")
        raise typer.Exit(code=1)
    else:
        out_dir.mkdir(parents=True)
        out_file = out_dir / "pipeline.py"
        contents = template_pipeline(short_name, pipeline)
        with open(out_file, "w") as f:
            f.write(contents)
        with open(out_dir / "requirements.txt", "w") as f:
            f.write("## Please specify all pipeline dependencies here\n")

        print(f"Generated pipeline scaffolding in {out_dir}.")

    return


@pipeline_app.command(no_args_is_help=True)
def register(
    pipeline_file: Path = typer.Argument(
        None,
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
        help=("Path to a Python file containing your pipeline implementation."),
    ),
):
    client = GardenClient()
    if (
        not pipeline_file.exists()
        or not pipeline_file.is_file()
        or pipeline_file.suffix != ".py"
    ):
        console.log(
            f"{pipeline_file} is not a valid Python file. Please provide a valid Python file (.py)."
        )
        raise typer.Exit(code=1)
    try:
        user_pipeline = load_pipeline_from_python_file(pipeline_file)
    except (
        PipelineLoadException,
        PipelineLoadScaffoldedException,
    ) as e:
        console.log(f"Could not parse {pipeline_file} as a Garden pipeline. " + str(e))
        raise typer.Exit(code=1) from e

    with console.status(
        "[bold green]Building container. This operation times out after 30 minutes."
    ):
        container_uuid = client.build_container(user_pipeline)
    console.print(f"Created container {container_uuid}")
    func_uuid = client.register_pipeline(user_pipeline, container_uuid)
    console.print(f"Created function {func_uuid}")
    console.print(f"Done! Pipeline was registered with doi {user_pipeline.doi}.")


@pipeline_app.command()
def shell(
    pipeline_file: Path = typer.Argument(
        None,
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
        help=("Path to a Python file containing your pipeline implementation."),
    ),
    requirements_file: Path = typer.Argument(
        None,
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
        help=("Path to a Python file containing your pipeline requirements."),
    ),
    env_name: Path = typer.Argument(
        None,
        dir_okay=False,
        file_okay=False,
        help=("The name to give your pipeline's virtual environment."),
    ),
):
    import os
    import subprocess
    import tempfile
    import sys

    try:
        # Create a virtual environment in the temp directory
        temp_dir = os.path.join(os.path.sep, tempfile.gettempdir(), env_name)
        print(f"Setting up environment in {temp_dir} ...")
        subprocess.run(["python3", "-m", "venv", temp_dir])

        # Activate the environment os dependent
        if sys.platform == "win32":
            activate_script = os.path.join(temp_dir, "Scripts", "activate.bat")
        elif sys.platform == "darwin":
            activate_script = os.path.join(temp_dir, "bin", "activate")
        else:
            activate_script = os.path.join(temp_dir, "bin", "activate")

        subprocess.run(f"source {activate_script}", shell=True)

        # Upgrade pip in the virtual environment quietly
        subprocess.run(
            f"{temp_dir}/bin/python3 -m pip install -q --upgrade pip",
            check=True,
            shell=True,
        )

        # Install the requirements with nice spinner
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Installing requirements...", total=None)

            subprocess.run(
                f"{temp_dir}/bin/pip install -q -r {requirements_file}",
                check=True,
                shell=True,
            )

        # Start Python shell in the virtual environment with the pipeline file
        print("Starting Garden test shell. Loading your pipeline one moment...")
        python_command = os.path.join(temp_dir, "bin", "python")
        subprocess.run([python_command, "-i", pipeline_file])

        # Clean up prompt for the temporary environment
        cleanup = typer.confirm(
            "Would you like to cleanup (delete) the virtual environment?"
        )
        if cleanup:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(
                    description="Removing up virtual environment...", total=None
                )
                import shutil

                shutil.rmtree(
                    temp_dir
                )  # Remove the directory listed under temp_dir not the actual tmp directory
        else:
            print(
                f"Virtual environment at {temp_dir} still remains and can be used for futher testing or manual removal."
            )

        print("Local Garden pipeline shell testing complete.")

    except Exception as e:
        # MVP error handling
        print(f"An error occurred: {e}")


@pipeline_app.command(no_args_is_help=False)
def list():
    """Lists all local pipelines."""

    resource_table_cols = ["doi", "title", "description"]
    table_name = "Local Pipelines"

    table = get_local_pipeline_rich_table(
        resource_table_cols=resource_table_cols, table_name=table_name
    )
    console.print("\n")
    console.print(table)


@pipeline_app.command(no_args_is_help=True)
def show(
    pipeline_ids: List[str] = typer.Argument(
        ...,
        help="The DOIs of the pipelines you want to show the local data for. "
        "e.g. ``pipeline show pipeline1_doi pipeline2_doi`` will show the local data for both pipelines listed.",
    ),
):
    """Shows all info for some Gardens"""

    for pipeline_id in pipeline_ids:
        pipeline = _get_pipeline(pipeline_id)
        if pipeline:
            rich.print(f"Pipeline: {pipeline_id} local data:")
            rich.print_json(json=pipeline.json())
            rich.print("\n")
        else:
            rich.print(f"Could not find pipeline with id {pipeline_id}\n")
