#!/usr/bin/env python3
# module for the bare "garden" command
import typer
import pathlib
from typing import List
import json
import datetime

app = typer.Typer()


@app.callback()
def help_info():
    """
    [friendly description of the garden CLI and/or project]

    maybe also some example usage?

    if we want to add opts for "bare garden" that'd come before any subcommand,
    here is where we'd declare them
    e.g. `garden [opts for "garden"] create [opts for "garden create"]`
    """
    pass


def is_valid_directory(directory: str):
    """
    validate the string optionally provided by the user as a directory for the
    garden.  should return the value if successful
    """
    pass


@app.command()
def create(
    directory: str = typer.Argument(
        pathlib.Path.cwd(),  # default to current directory
        callback=is_valid_directory,
    ),
    title: str = typer.Option(
        ...,
        "-t",
        "--title",
        prompt="Enter a title for your Garden:",
        help="The Garden's official title (as should appear in citations)",
        rich_help_panel="Required",
    ),
    authors: List[str] = typer.Option(
        ...,
        "-a",
        "--authors",
        prompt='Enter an author ("Family, Given"):',
        help="This can be repeated for multiple authors (order is preserved).",
        rich_help_panel="Required",
    ),
    description: str = typer.Option(
        None,
        help=(
            "A brief summary of the Garden and/or its purpose, to aid "
            "discovery by other Gardeners."
        ),
        rich_help_panel="Recommended",
    ),
):
    """Create a Garden entity."""

    return

    # probably want `garden add model` to prompt with a list of known pipelines
    # default to reading from garden.json if one is found?
