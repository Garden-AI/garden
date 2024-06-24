import logging
import typer

logger = logging.getLogger()

hpc_notebook_app = typer.Typer(name="hpc-notebook")


@hpc_notebook_app.callback(no_args_is_help=True)
def hpc_notebook():
    """sub-commands for editing and publishing from sandboxed notebooks in HPC."""
    pass


@hpc_notebook_app.command()
def start():
    """Open a notebook file in HPC."""
    print("🚧🌱🚧 Under Construction 🚧🌱🚧")


@hpc_notebook_app.command()
def publish():
    """Publish your hpc-notebook."""
    print("🚧🌱🚧 Under Construction 🚧🌱🚧")
