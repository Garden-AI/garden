import logging
from typing import List

import typer
import rich
from rich.prompt import Prompt

from garden_ai.app.console import console, get_local_entrypoint_rich_table
from garden_ai.app.completion import complete_entrypoint
from garden_ai.entrypoints import Repository, Paper
from garden_ai.local_data import (
    put_local_entrypoint,
    get_local_entrypoint_by_doi,
)


logger = logging.getLogger()

entrypoint_app = typer.Typer(name="entrypoint", no_args_is_help=True)


def parse_full_name(name: str) -> str:
    """(this will probably eventually use some 3rd party name parsing library)"""
    return name.strip() if name else ""


@entrypoint_app.callback()
def entrypoint():
    """
    sub-commands for creating and manipulating entrypoints
    """
    pass


@entrypoint_app.command(no_args_is_help=True)
def add_repository(
    doi: str = typer.Option(
        ...,
        "-d",
        "--doi",
        autocompletion=complete_entrypoint,
        help="The DOI for the entrypoint you would like to add a repository to",
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
    # get registered entrypoint
    entrypoint = get_local_entrypoint_by_doi(doi)
    if not entrypoint:
        rich.print(f"Could not find entrypoint with id {doi}\n")
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
        entrypoint.repositories.append(repository)
        put_local_entrypoint(entrypoint)
        rich.print(f"Repository added to entrypoint {doi}.")


@entrypoint_app.command(no_args_is_help=True)
def add_paper(
    doi: str = typer.Option(
        ...,
        "-d",
        "--doi",
        autocompletion=complete_entrypoint,
        help="The DOI for the entrypoint you would like to link a paper to",
        rich_help_panel="Required",
    ),
    title: str = typer.Option(
        ...,
        "-t",
        "--title",
        prompt=("The title of the paper you would like to add to your entrypoint"),
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
    entrypoint = get_local_entrypoint_by_doi(doi)
    if not entrypoint:
        rich.print(f"Could not find entrypoint with id {doi}\n")
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
        entrypoint.papers.append(paper)
        put_local_entrypoint(entrypoint)
        rich.print(f"The paper {title} is successfully added to entrypoint {doi}.")


@entrypoint_app.command(no_args_is_help=False)
def list():
    """Lists all local entrypoints."""

    resource_table_cols = ["doi", "title", "description"]
    table_name = "Local Entrypoints"

    table = get_local_entrypoint_rich_table(
        resource_table_cols=resource_table_cols, table_name=table_name
    )
    console.print("\n")
    console.print(table)


@entrypoint_app.command(no_args_is_help=True)
def show(
    entrypoint_ids: List[str] = typer.Argument(
        ...,
        help="The DOIs of the entrypoints you want to show local data for. ",
        autocompletion=complete_entrypoint,
    ),
):
    """Shows all info for some entrypoints"""

    for entrypoint_id in entrypoint_ids:
        entrypoint = get_local_entrypoint_by_doi(entrypoint_id)
        if entrypoint:
            rich.print(f"Entrypoint: {entrypoint_id} local data:")
            rich.print_json(json=entrypoint.json())
            rich.print("\n")
        else:
            rich.print(f"Could not find entrypoint with id {entrypoint_id}")
