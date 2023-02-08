#!/usr/bin/env python3
# module for the bare "garden" command
import typer
import pathlib
import time
import rich
from typing import List, Optional
from rich import print
from rich.prompt import Prompt
from datetime import datetime

import logging
from garden_ai.client import GroupsClient, SearchClient, GardenClient, AuthAPIError

from pathlib import Path

logger = logging.getLogger()

app = typer.Typer()


@app.callback()
def help_info():
    """
    [friendly description of the garden CLI and/or project]

    maybe also some example usage? This docstring is automatically turned into --help text.

    if we want to add opts for "bare garden" that'd come before any subcommand,
    here is where we'd declare them e.g. `garden [opts for "garden"] create
    [opts for "garden create"]`
    """
    pass


def setup_directory(directory: Path) -> Path:
    """
    Validate the directory provided by the user, scaffolding with "pipelines/" and
    "models/" subdirectories if possible (i.e. directory does not yet exist or
    exists but is empty).
    """
    if directory.exists():
        if list(directory.iterdir()):
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
    """ """
    return name.strip() if name else ""


def cli_do_login_flow(self: GardenClient):
    """
    drop-in replacement for `_do_login_flow` that uses typer/click helper
    functions to launch the globus auth url automatically, so users don't have to
    copy a url from their terminal.
    """
    self.auth_client.oauth2_start_flow(
        requested_scopes=[
            GroupsClient.scopes.view_my_groups_and_memberships,
            SearchClient.scopes.ingest,
            GardenClient.scopes.action_all,  # "https://auth.globus.org/scopes/0948a6b0-a622-4078-b0a4-bfd6d77d65cf/action_all"
        ],
        refresh_tokens=True,
    )
    authorize_url = self.auth_client.oauth2_get_authorize_url()
    print(
        f"Authenticating with Globus in your default web browser: \n\n{authorize_url}"
    )
    time.sleep(3)
    typer.launch(authorize_url)

    auth_code = Prompt.ask("Please enter the code here ").strip()

    try:
        tokens = self.auth_client.oauth2_exchange_code_for_tokens(auth_code)
        return tokens
    except AuthAPIError:
        logger.fatal("Invalid Globus auth token received. Exiting")
        raise typer.Exit(code=1)


# replace login flow method used by GardenClient:
GardenClient._do_login_flow = cli_do_login_flow
# ^I feel like this isn't good practice, but I'm not sure it's worth trying to
# get the typer session/prompting behavior in the sdk client module when the
# sdk doesn't need to know about the CLI for any other reason


@app.command()
def create(
    directory: Path = typer.Argument(
        pathlib.Path.cwd(),  # default to current directory
        callback=setup_directory,  # TODO
        dir_okay=True,
        file_okay=False,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
    authors: List[str] = typer.Option(
        None,
        "-a",
        "--author",
        help=(
            "Name an author of this Garden. Repeat this to indicate multiple authors: "
            "`garden create ... --author='Mendel, Gregor' -a 'Other-Author, Anne' ...` (order is preserved)."
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
            "Provide a brief description of this Garden to aid in discovery (leave blank to skip)"
        )

    client = GardenClient()
    garden = client.create_garden(
        authors=authors,
        title=title,
        year=year,
        description=description,
        contributors=contributors,
    )
    # TODO just until doi minting via backend is demo-ready
    garden.doi = "10.26311/fake-doi"
    client.register_metadata(garden, directory)  # writes garden.json

    with open(directory / "garden.json", "r") as f_in:
        metadata = f_in.read()
        rich.print_json(metadata)

    return
