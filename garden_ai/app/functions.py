"""CLI commands for function management (Modal + HPC)."""

import typer

from garden_ai.app.groundhog import groundhog_app
from garden_ai.app.modal_cmds import modal_app

functions_app = typer.Typer(
    help="Manage functions (Modal and HPC)", no_args_is_help=True
)
functions_app.add_typer(modal_app, name="modal")
functions_app.add_typer(groundhog_app, name="hpc")
