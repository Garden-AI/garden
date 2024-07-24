import logging
import typer
import os
import subprocess
import tempfile
import sys

logger = logging.getLogger()

hpc_notebook_app = typer.Typer(name="hpc-notebook")


@hpc_notebook_app.callback(no_args_is_help=True)
def hpc_notebook():
    """sub-commands for editing and publishing from sandboxed notebooks in HPC."""
    pass


def start_apptainer(notebooks_dir, container_image):
    process = None
    try:
        with tempfile.TemporaryDirectory() as working_directory:
            tmp_dir = os.path.join(working_directory, "tmp")
            os.makedirs(tmp_dir, exist_ok=True)

            # Step 2: Set environment variables
            os.environ["SINGULARITY_TMPDIR"] = tmp_dir
            os.environ["APPTAINER_TMPDIR"] = tmp_dir
            os.environ["JUPYTER_RUNTIME_DIR"] = tmp_dir
            os.environ["JUPYTER_DATA_DIR"] = tmp_dir
            os.environ["JUPYTER_CONFIG_DIR"] = tmp_dir

            run_command = [
                "apptainer",
                "run",
                "--bind",
                f"{notebooks_dir}:/notebooks",
                container_image,
                "jupyter",
                "notebook",
                "--no-browser",
                "--ip=0.0.0.0",
            ]
            process = subprocess.Popen(run_command)
            process.wait()
            logger.info(
                "Jupyter Notebook started successfully in the Apptainer container."
            )
    except KeyboardInterrupt:
        if process:
            process.terminate()
            process.wait()
        logger.info("Operation cancelled by user (Ctrl-C).")
        typer.echo("Operation cancelled by user (Ctrl-C).")
        sys.exit(0)


@hpc_notebook_app.command()
def rerun(container_image: str = "hpc-notebook.sif"):
    current_directory = os.getcwd()
    notebooks_dir = os.path.join(current_directory, "notebooks")

    if not os.path.exists(container_image):
        logger.error("Not found.")
        typer.echo("Not found")
    else:
        start_apptainer(notebooks_dir, container_image)


@hpc_notebook_app.command()
def start(container_image: str = "hpc-notebook.sif"):
    """Open a notebook file in HPC."""

    # Get the absolute path of the definition file located three levels up from the current script's directory
    definition_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../scripts/Singularity.def")
    )

    current_directory = os.getcwd()
    notebooks_dir = os.path.join(current_directory, "notebooks")

    os.makedirs(notebooks_dir, exist_ok=True)

    try:
        # Ensure the definition file exists
        if not os.path.isfile(definition_file):
            logger.error(f"Definition file {definition_file} not found.")
            typer.echo(f"🚧🌱🚧 Definition file {definition_file} not found 🚧🌱🚧")
            raise typer.Exit(code=1)

        # Step 3: Build the Apptainer container
        build_command = ["apptainer", "build", container_image, definition_file]
        subprocess.run(build_command, check=True)
        logger.info("Apptainer container built successfully.")

        # Step 4: Run the Apptainer container and start Jupyter Notebook
        start_apptainer(notebooks_dir, container_image)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Jupyter Notebook: {e}")
        typer.echo("🚧🌱🚧 Failed to start Jupyter Notebook 🚧🌱🚧")
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user (Ctrl-C).")
        typer.echo("Operation cancelled by user (Ctrl-C).")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        typer.echo("🚧🌱🚧 An unexpected error occurred 🚧🌱🚧")


@hpc_notebook_app.command()
def publish():
    """Publish your hpc-notebook."""
    print("🚧🌱🚧 Under Construction 🚧🌱🚧")


if __name__ == "__main__":
    hpc_notebook_app()
