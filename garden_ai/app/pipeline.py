import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
from tempfile import TemporaryDirectory, NamedTemporaryFile

import dill  # type: ignore
import jinja2
import typer
import rich
from rich import print
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn

from garden_ai import GardenClient, Pipeline, step, GardenConstants
from garden_ai.app.console import console, get_local_pipeline_rich_table
from garden_ai.app.completion import complete_pipeline
from garden_ai.pipelines import RegisteredPipeline, Repository, Paper
from garden_ai.local_data import (
    _read_local_cache,
    put_local_pipeline,
    get_local_pipeline_by_doi,
)
from garden_ai.mlmodel import PipelineLoadScaffoldedException
from garden_ai.utils._meta import make_func_to_serialize
from garden_ai.utils.filesystem import (
    load_pipeline_from_python_file,
    PipelineLoadException,
)
from garden_ai.utils.misc import clean_identifier, get_cache_tag


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
def add_repository(
    doi: str = typer.Option(
        ...,
        "-d",
        "--doi",
        autocompletion=complete_pipeline,
        help="The DOI for the pipeline you would like to add a repository to",
        rich_help_panel="Required",
    ),
    url: str = typer.Option(
        ...,
        "-u",
        "--url",
        prompt="The url which the repository can be accessed in",
        rich_help_panel="Required",
    ),
    repository_name: str = typer.Option(
        ...,
        "-r",
        "--repository_name",
        prompt=("The name of your repository"),
        rich_help_panel="Required",
    ),
    contributors: List[str] = typer.Option(
        None,
        "-c",
        "--contributor",
        help=(
            "Acknowledge a contributor in this repository. Repeat to indicate multiple (like --author)."
        ),
        rich_help_panel="Recommended",
    ),
):
    # get registered pipeline
    pipeline = get_local_pipeline_by_doi(doi)
    if not pipeline:
        rich.print(f"Could not find pipeline with id {doi}\n")
    else:
        if not contributors:
            name = parse_full_name(
                Prompt.ask("Acknowledge a contributor? (leave blank to skip)")
            )
            if name:
                contributors = [name]
                while True:
                    name = parse_full_name(
                        Prompt.ask("Add another contributor? (leave blank to finish)")
                    )
                    if name:
                        contributors += [name]
                    else:
                        break
        repository = Repository(
            repo_name=repository_name, url=url, contributors=contributors
        )
        pipeline.repositories.append(repository)
        put_local_pipeline(pipeline)
        rich.print(f"Repository added to pipeline {doi}.")


@pipeline_app.command(no_args_is_help=True)
def add_paper(
    doi: str = typer.Option(
        ...,
        "-d",
        "--doi",
        autocompletion=complete_pipeline,
        help="The DOI for the pipeline you would like to add a repository to",
        rich_help_panel="Required",
    ),
    title: str = typer.Option(
        ...,
        "-t",
        "--title",
        prompt=("The title of the paper you would like to add to your pipeline"),
        rich_help_panel="Required",
    ),
    authors: List[str] = typer.Option(
        None,
        "-a",
        "--author",
        help=(
            "Acknowledge an author in this repository. Repeat to indicate multiple (like --author)."
        ),
        rich_help_panel="Recommended",
    ),
    paper_doi: str = typer.Option(
        None,
        "-p",
        "--paper-doi",
        help=("Optional, the digital identifier that the paper may be linked to"),
        rich_help_panel="Recommended",
    ),
    citation: str = typer.Option(
        None,
        "-c",
        "--citation",
        help=("Optional, enter how the paper may be cited."),
        rich_help_panel="Recommended",
    ),
):
    pipeline = get_local_pipeline_by_doi(doi)
    if not pipeline:
        rich.print(f"Could not find pipeline with id {doi}\n")
    else:
        if not authors:
            name = parse_full_name(
                Prompt.ask("Acknowledge an author? (leave blank to skip)")
            )
            if name:
                authors = [name]
                while True:
                    name = parse_full_name(
                        Prompt.ask("Add another author? (leave blank to finish)")
                    )
                    if name:
                        authors += [name]
                    else:
                        break
        if not paper_doi:
            paper_doi = Prompt.ask(
                "If available, please provite the digital identifier that the paper may be linked to (leave blank to skip)"
            )
        if not citation:
            citation = Prompt.ask(
                "If available, please provite the citation that the paper may be linked to (leave blank to skip)"
            )
        paper = Paper(title=title, authors=authors, doi=paper_doi, citation=citation)
        pipeline.papers.append(paper)
        put_local_pipeline(pipeline)
        rich.print(f"The paper {title} is successfully added to pipeline {doi}.")


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
        raise
    with console.status(
        "[bold green]Building container and registering pipeline. This operation times out after 30 minutes."
    ):
        client.register_pipeline(user_pipeline)
    console.print(f"Done! Pipeline was registered with DOI {user_pipeline.doi}.")


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


@pipeline_app.command()
def container(
    container_uuid: str = typer.Option(
        None,
        "-u",
        "--uuid",
        help="The UUID of the container you would like to shell into, considered first when multiple options are provided",
    ),
    doi: str = typer.Option(
        None,
        "-d",
        "--doi",
        autocompletion=complete_pipeline,
        help="The DOI for the pipeline whose container you would like to shell into, considered second when multiple options are provided",
    ),
    pipeline_file: Path = typer.Option(
        None,
        "-p",
        "--pipeline-file",
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
        help=(
            "Path to a Python file containing your pipeline implementation, considered third when multiple options are provided"
        ),
    ),
    cleanup: bool = typer.Option(
        False,
        "--rm",
        help="When set, the downloaded container image will be removed automatically",
    ),
):
    import subprocess

    client = GardenClient()

    if container_uuid:
        container_info = client.compute_client.get_container(container_uuid, "docker")
    elif doi:
        user_pipeline: Union[
            Pipeline, RegisteredPipeline
        ] = client.get_registered_pipeline(doi)

        cached_uuid = _read_local_cache().get(
            get_cache_tag(
                user_pipeline.pip_dependencies,
                user_pipeline.conda_dependencies,
                user_pipeline.python_version,
            )
        )
        if cached_uuid is None:
            with console.status(
                "[bold green]Building container. This operation times out after 30 minutes."
            ):
                cached_uuid = client.build_container(user_pipeline)

        container_info = client.compute_client.get_container(cached_uuid, "docker")
    elif pipeline_file:
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
            console.log(
                f"Could not parse {pipeline_file} as a Garden pipeline. " + str(e)
            )
            raise
        cached_uuid = _read_local_cache().get(
            get_cache_tag(
                user_pipeline.pip_dependencies,
                user_pipeline.conda_dependencies,
                user_pipeline.python_version,
            )
        )
        if cached_uuid is None:
            with console.status(
                "[bold green]Building container. This operation times out after 30 minutes."
            ):
                cached_uuid = client.build_container(user_pipeline)

        container_info = client.compute_client.get_container(cached_uuid, "docker")
    else:
        console.log(
            "A container's UUID, the DOI of a local pipeline, or a pipeline file must be specified"
        )
        raise typer.Exit(code=1)

    location = container_info["location"]

    subprocess.run(["docker", "pull", location])
    subprocess.run(["docker", "run", "-it", "--rm", "--entrypoint", "bash", location])
    if cleanup:
        subprocess.run(["docker", "rmi", location])


@pipeline_app.command(no_args_is_help=True)
def debug(
    pipeline_file: Path = typer.Argument(
        None,
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
        help=("Path to a Python file containing your pipeline implementation."),
    ),
    docker: bool = typer.Option(
        False,
        "--docker",
        help="When set, the pipeline session will be loaded within an automatically built container",
    ),
    cleanup: bool = typer.Option(
        False,
        "--rm",
        help="When set, the built container image will be removed automatically (only used when --docker is also supplied)",
    ),
):
    import subprocess
    from functools import partial

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
        pipeline_url_json = client.generate_presigned_urls_for_pipeline(user_pipeline)
        _env_vars = {GardenConstants.URL_ENV_VAR_NAME: pipeline_url_json}
    except (
        PipelineLoadException,
        PipelineLoadScaffoldedException,
    ) as e:
        console.log(f"Could not parse {pipeline_file} as a Garden pipeline. " + str(e))
        raise
    remote_func = make_func_to_serialize(user_pipeline)
    curry = partial(remote_func, _env_vars=_env_vars)

    with TemporaryDirectory() as tmpdir:
        tmpdir = tmpdir.replace("\\", "/")  # Windows compatibility
        with NamedTemporaryFile(dir=tmpdir, delete=False) as tmp:
            dill.dump(curry, tmp)
            tmp_filename = Path(tmp.name).name

        # contains everything, control flow dictates which pieces are removed
        interpreter_cmd = [
            "python",
            "-i",
            "-c",
            f'\'import dill; f = open("{tmpdir}/{tmp_filename}", "rb"); {user_pipeline.short_name} = dill.load(f); f.close(); del f; del dill; '
            f'print(f"Defined your pipeline as the function `{user_pipeline.short_name}`!")\'',
        ]

        if docker:
            image_name = f"gardenai/{user_pipeline.short_name}"
            # avoids SyntaxError in inline version
            escaped_pip_deps = [f'"{dep}"' for dep in user_pipeline.pip_dependencies]
            # remove tmpdir in container's path
            interpreter_cmd[-1] = interpreter_cmd[-1].replace(f"{tmpdir}/", "")

            with open(f"{tmpdir}/Dockerfile", "w+") as dockerfile:
                dockerfile.write(
                    "FROM continuumio/miniconda3\n"
                    f'COPY "{tmp_filename}" .\n'
                    f"RUN conda create -y -n env pip python={user_pipeline.python_version}\n"
                    'SHELL ["conda", "run", "--no-capture-output", "-n", "env", "/bin/bash", "-c"]\n'
                )
                if user_pipeline.conda_dependencies:
                    dockerfile.write(
                        f"RUN conda install -y {' '.join(user_pipeline.conda_dependencies)}\n"
                    )
                dockerfile.write(
                    f"RUN pip install --no-build-isolation dill {' '.join(escaped_pip_deps)}\n"
                    f"ENTRYPOINT {' '.join(interpreter_cmd)}"
                )

            subprocess.run(["docker", "build", "-t", image_name, tmpdir])
            subprocess.run(["docker", "run", "-it", "--rm", image_name])
            if cleanup:
                subprocess.run(["docker", "rmi", image_name])
        else:
            if os.name == "nt":
                # remove encapsulating quotes bash needs
                interpreter_cmd[-1] = interpreter_cmd[-1].replace("'", "")
            subprocess.run(interpreter_cmd)


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
        autocompletion=complete_pipeline,
    ),
):
    """Shows all info for some Gardens"""

    for pipeline_id in pipeline_ids:
        pipeline = get_local_pipeline_by_doi(pipeline_id)
        if pipeline:
            rich.print(f"Pipeline: {pipeline_id} local data:")
            rich.print_json(json=pipeline.json())
            rich.print("\n")
        else:
            rich.print(f"Could not find pipeline with id {pipeline_id}")
