import logging
from typing import List

import rich
import typer
from rich.prompt import Prompt

from garden_ai import GardenClient
from garden_ai.app.completion import complete_entrypoint
from garden_ai.app.console import console, get_owned_entrypoints_rich_table
from garden_ai.schemas.entrypoint import PaperMetadata, RepositoryMetadata
from garden_ai.utils.interactive_cli import gui_edit_garden_entity

logger = logging.getLogger()

entrypoint_app = typer.Typer(name="entrypoint", no_args_is_help=True)


def parse_full_name(name: str) -> str:
    """(this may eventually use some 3rd party name parsing library)"""
    return name.strip() if name else ""


@entrypoint_app.callback()
def entrypoint():
    """
    sub-commands for creating and manipulating entrypoints
    """
    pass


@entrypoint_app.command(no_args_is_help=True)
def add_repository(
    doi: str = typer.Argument(
        ...,
        shell_complete=complete_entrypoint,
        help="The DOI for the entrypoint you would like to link to a code repository",
        rich_help_panel="Required",
    ),
    url: str = typer.Option(
        ...,
        "-u",
        "--url",
        prompt="The url linking to the repository",
        rich_help_panel="Required",
    ),
    repository_name: str = typer.Option(
        ...,
        "-n",
        "--name",
        prompt=("The name to display for your repository"),
        rich_help_panel="Required",
    ),
    contributors: List[str] = typer.Option(
        None,
        "-c",
        "--contributor",
        help=(
            "Acknowledge a contributor to this repository. Repeat to indicate multiple."
        ),
        rich_help_panel="Recommended",
    ),
):
    client = GardenClient()
    entrypoint_meta = client.backend_client.get_entrypoint_metadata(doi)

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
    repo_meta = RepositoryMetadata(
        repo_name=repository_name, url=url, contributors=contributors
    )
    entrypoint_meta.repositories.append(repo_meta)
    client.backend_client.put_entrypoint_metadata(entrypoint_meta)
    rich.print(
        f"Repository {repository_name} linked to entrypoint {entrypoint_meta.doi}."
    )


@entrypoint_app.command(no_args_is_help=True)
def add_paper(
    entrypoint_doi: str = typer.Argument(
        ...,
        shell_complete=complete_entrypoint,
        help="The DOI of the entrypoint you would like to link to a paper",
        rich_help_panel="Required",
    ),
    title: str = typer.Option(
        ...,
        "-t",
        "--title",
        prompt=("The title of the paper to link"),
        rich_help_panel="Required",
    ),
    authors: List[str] = typer.Option(
        None,
        "-a",
        "--author",
        help=("Add an author of the paper. Repeat to indicate multiple."),
        rich_help_panel="Recommended",
    ),
    paper_doi: str = typer.Option(
        None,
        "-d",
        "--doi",
        help=("Optional, the DOI of the paper to be linked"),
        rich_help_panel="Recommended",
    ),
    citation: str = typer.Option(
        None,
        "-c",
        "--citation",
        help=("Optional, a citation to display for the linked paper."),
        rich_help_panel="Recommended",
    ),
):
    client = GardenClient()
    entrypoint_meta = client.backend_client.get_entrypoint_metadata(entrypoint_doi)
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
            "If available, please provite the DOI that the paper may be linked to (leave blank to skip)"
        )
    if not citation:
        citation = Prompt.ask(
            "If available, please provide an official citation to display for the paper (leave blank to skip)"
        )
    paper = PaperMetadata(
        title=title, authors=authors, doi=paper_doi, citation=citation
    )
    entrypoint_meta.papers.append(paper)
    client.backend_client.put_entrypoint_metadata(entrypoint_meta)
    rich.print(
        f"The paper {title} was successfully added to entrypoint {entrypoint_doi}."
    )


@entrypoint_app.command(no_args_is_help=True)
def register_doi(
    doi: str = typer.Argument(
        ...,
        shell_complete=complete_entrypoint,
        help="The draft entrypoint DOI you want to register",
        rich_help_panel="Required",
    ),
):
    """
    Publicly register an Entrypoint's DOI, moving it out of draft state.

    NOTE: Entrypoints with registered DOIs cannot be deleted.
    """
    client = GardenClient()
    client.register_entrypoint_doi(doi)
    rich.print(f"DOI {doi} has been moved out of draft status and can now be cited.")


@entrypoint_app.command(no_args_is_help=False)
def list():
    """Lists all owned entrypoints."""

    client = GardenClient()
    resource_table_cols = ["doi", "title", "description", "doi_is_draft"]
    table_name = "My Entrypoints"

    table = get_owned_entrypoints_rich_table(
        client, resource_table_cols=resource_table_cols, table_name=table_name
    )
    console.print("\n")
    console.print(table)


@entrypoint_app.command(no_args_is_help=True)
def show(
    entrypoint_ids: List[str] = typer.Argument(
        ...,
        help="The DOIs of the entrypoints you want to show local data for. ",
        shell_complete=complete_entrypoint,
    ),
):
    """Shows all info for some entrypoints"""
    client = GardenClient()
    entrypoints = client.backend_client.get_entrypoints(dois=entrypoint_ids)

    for entrypoint in entrypoints:
        rich.print(f"Entrypoint {entrypoint.metadata.doi} data:")
        rich.print_json(json=entrypoint.metadata.model_dump_json())
        rich.print("\n")


@entrypoint_app.command()
def edit(
    doi: str = typer.Argument(
        ...,
        shell_complete=complete_entrypoint,
        help="The DOI of the entrypoint you want to edit",
        rich_help_panel="Required",
    )
):
    """Edit an Entrypoint's metadata"""
    client = GardenClient()
    entrypoint_meta = client.backend_client.get_entrypoint_metadata(doi)

    string_fields = ["title", "description", "year", "short_name"]
    list_fields = ["authors", "tags"]

    edited_entrypoint_meta = gui_edit_garden_entity(
        entrypoint_meta, string_fields, list_fields
    )
    client.backend_client.put_entrypoint_metadata(edited_entrypoint_meta)

    console.print(f"Updated entrypoint {doi}.")
