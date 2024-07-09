import logging
from typing import List

import typer
import rich
from rich.prompt import Prompt

from garden_ai import GardenClient
from garden_ai.app.console import (
    console,
    get_local_entrypoint_rich_table,
    DOI_STATUS_COLUMN,
)
from garden_ai.utils.interactive_cli import gui_edit_garden_entity
from garden_ai.app.completion import complete_entrypoint
from garden_ai.entrypoints import Repository, Paper
from garden_ai import local_data
from garden_ai.local_data import (
    put_local_entrypoint,
    get_local_entrypoint_by_doi,
)


logger = logging.getLogger()

entrypoint_app = typer.Typer(name="entrypoint", no_args_is_help=True)


def _get_entrypoint(doi):
    if local_data._IS_DISABLED:
        client = GardenClient()
        entrypoint = client.backend_client.get_entrypoint(doi)
    else:
        entrypoint = get_local_entrypoint_by_doi(doi)
    return entrypoint


def _put_entrypoint(entrypoint):
    if local_data._IS_DISABLED:
        client = GardenClient()
        client.backend_client.update_entrypoint(entrypoint)
    else:
        put_local_entrypoint(entrypoint)
    return


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
    entrypoint = _get_entrypoint(doi)

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
        _put_entrypoint(entrypoint)
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
    entrypoint = _get_entrypoint(doi)
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
        _put_entrypoint(entrypoint)
        rich.print(f"The paper {title} is successfully added to entrypoint {doi}.")


@entrypoint_app.command(no_args_is_help=True)
def register_doi(
    doi: str = typer.Argument(
        ...,
        autocompletion=complete_entrypoint,
        help="The draft entrypoint DOI you want to register",
        rich_help_panel="Required",
    ),
):
    """
    Moves an Entrypoint's DOI out of draft state.

    Parameters
    ----------
    doi : str
        The DOI of the entrypoint to be registered.
    """
    client = GardenClient()
    entrypoint = _get_entrypoint(doi)
    if not entrypoint:
        rich.print(f"Could not find entrypoint with doi {doi}")
        raise typer.Exit(code=1)
    client.register_entrypoint_doi(entrypoint)
    _put_entrypoint(entrypoint)
    rich.print(f"DOI {doi} has been moved out of draft status and can now be cited.")


if not local_data._IS_DISABLED:
    # this subcommand is no longer meaningful when local_data is disabled.
    # we can replace this with "list my gardens" behavior once
    # https://github.com/Garden-AI/garden-backend/issues/111 is live.
    @entrypoint_app.command(no_args_is_help=False)
    def list():
        """Lists all local entrypoints."""

        resource_table_cols = ["doi", "title", "description", DOI_STATUS_COLUMN]
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
        entrypoint = _get_entrypoint(entrypoint_id)
        if entrypoint:
            rich.print(f"Entrypoint: {entrypoint_id} local data:")
            rich.print_json(json=entrypoint.model_dump_json())
            rich.print("\n")
        else:
            rich.print(f"Could not find entrypoint with id {entrypoint_id}")


@entrypoint_app.command()
def edit(
    doi: str = typer.Argument(
        ...,
        autocompletion=complete_entrypoint,
        help="The DOI of the entrypoint you want to edit",
        rich_help_panel="Required",
    )
):
    """Edit an Entrypoint's metadata"""

    entrypoint = _get_entrypoint(doi)
    if not entrypoint:
        rich.print(f"Could not find entrypoint with doi {doi}")
        raise typer.Exit(code=1)

    string_fields = ["title", "description", "year", "short_name"]
    list_fields = ["authors", "tags"]

    edited_entrypoint = gui_edit_garden_entity(entrypoint, string_fields, list_fields)

    _put_entrypoint(edited_entrypoint)
    console.print(
        "Updated entrypoint {doi}. For the changes to be reflected on thegardens.ai, publish a garden that this entrypoint belongs to."
    )
