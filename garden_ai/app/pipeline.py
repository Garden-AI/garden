import logging
from typing import List

import typer
import rich
from rich.prompt import Prompt

from garden_ai.app.console import console, get_local_pipeline_rich_table
from garden_ai.app.completion import complete_pipeline
from garden_ai.pipelines import Repository, Paper
from garden_ai.local_data import (
    put_local_pipeline,
    get_local_pipeline_by_doi,
)


logger = logging.getLogger()

pipeline_app = typer.Typer(name="pipeline", no_args_is_help=True)


def parse_full_name(name: str) -> str:
    """(this will probably eventually use some 3rd party name parsing library)"""
    return name.strip() if name else ""


@pipeline_app.callback()
def pipeline():
    """
    sub-commands for creating and manipulating Pipelines
    """
    pass


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
