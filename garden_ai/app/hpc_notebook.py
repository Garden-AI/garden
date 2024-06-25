import logging
import shutil
import typer
from pathlib import Path
from typing import Optional

from garden_ai import GardenConstants
from garden_ai.app.notebook import _get_base_image_uri

from garden_ai.notebook_metadata import (
    add_notebook_metadata,
    get_notebook_metadata,
    set_notebook_metadata,
    read_requirements_data,
)
from garden_ai.utils.notebooks import (
    generate_botanical_filename,
    _validate_requirements_path,
)

logger = logging.getLogger()

hpc_notebook_app = typer.Typer(name="hpc-notebook")


@hpc_notebook_app.callback(no_args_is_help=True)
def hpc_notebook():
    """sub-commands for editing and publishing from sandboxed notebooks in HPC."""
    pass


@hpc_notebook_app.command(no_args_is_help=True)
def start(
    path: Optional[Path] = typer.Argument(
        default=None,
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
        help=("Path to a .ipynb notebook to open in a HPC container."),
    ),
    base_image_name: Optional[str] = typer.Option(
        None,
        "--base-image",
        help=(
            "A Garden base image to boot the notebook in. "
            "For example, to boot your notebook with the default Garden python 3.8 image, use --base-image 3.8-base. "
        ),
    ),
    requirements_path: Optional[Path] = typer.Option(
        None,
        "--requirements",
        help=(
            "Path to a requirements.txt containing "
            "additional dependencies to install in the base image."
        ),
    ),
    global_notebook_doi: Optional[str] = typer.Option(
        None,
        "--doi",
        help=(
            "DOI of a Garden you want to add all entrypoints in this notebook too. "
            "To override the global notebook DOI for a specific entrypoint, "
            "provide the garden_entrypoint decorator with the optional garden_doi argument."
        ),
    ),
    custom_image_uri: Optional[str] = typer.Option(
        None,
        "--custom-image",
        help=(
            "Power users only! Provide a uri of a publicly available docker image to boot the notebook in."
        ),
        hidden=True,
    ),
):
    """Open a notebook file in HPC."""
    # First figure out the name of the notebook and whether we need to create it
    need_to_create_notebook = False

    if path is None:
        need_to_create_notebook = True
        new_notebook_name = generate_botanical_filename()
        notebook_path = Path.cwd() / new_notebook_name
    else:
        notebook_path = path.resolve()
        if notebook_path.suffix != ".ipynb":
            typer.echo("File must be a jupyter notebook (.ipynb)")
            raise typer.Exit(1)

        if not notebook_path.exists():
            need_to_create_notebook = True

    # Figure out what base image uri we should start the notebook in
    base_image_uri = _get_base_image_uri(
        base_image_name,
        custom_image_uri,
        None if need_to_create_notebook else notebook_path,
    )

    # Build prompt
    if need_to_create_notebook:
        message = f"This will create a new notebook {notebook_path.name} and open it in Docker image {base_image_uri}.\n"
    else:
        message = f"This will open existing notebook {notebook_path.name} in Docker image {base_image_uri}.\n"

    if requirements_path:
        message += f"Additional dependencies specified in {requirements_path.name} will also be installed in {base_image_uri}.\n"
        message += "Any dependencies previously associated with this notebook will be overwritten by the new requirements.\n"

    # Now we have all we need to prompt the user to proceed
    typer.confirm(message + "Do you want to proceed?", abort=True)

    if need_to_create_notebook:
        if base_image_name:
            template_file_name = GardenConstants.IMAGES_TO_FLAVOR.get(
                base_image_name, "empty.ipynb"
            )
        else:
            template_file_name = "empty.ipynb"

        top_level_dir = Path(__file__).parent.parent
        source_path = top_level_dir / "notebook_templates" / template_file_name
        shutil.copy(source_path, notebook_path)

    # Adds metadata widget cell and garden_metadata dict if either is missing
    add_notebook_metadata(notebook_path)

    notebook_metadata = get_notebook_metadata(notebook_path)

    # Validate and read requirements file.
    if requirements_path:
        requirements_path = requirements_path.resolve()
        _validate_requirements_path(requirements_path)
        requirements_data = read_requirements_data(requirements_path)
    else:
        # If no requirements file given, look for requirements data in notebook metadata
        requirements_data = notebook_metadata.notebook_requirements

    # Check for base image name from notebook if user did not provide one.
    if base_image_name is None:
        # If a user is using a custom image uri, base_image_name might be None.
        base_image_name = notebook_metadata.notebook_image_name

    # Check for global notebook doi from notebook if user did not provide one.
    if global_notebook_doi is None:
        global_notebook_doi = notebook_metadata.global_notebook_doi

    # Update garden metadata in notebook
    set_notebook_metadata(
        notebook_path,
        global_notebook_doi,
        base_image_name,
        base_image_uri,
        requirements_data,
    )

    print(
        f"Starting hpc-notebook inside base image with full name {base_image_uri}. "
        f"If you start this notebook again from the same folder, it will use this image by default."
    )

    print("🚧🌱🚧 Under Construction 🚧🌱🚧")


@hpc_notebook_app.command()
def publish():
    """Publish your hpc-notebook."""
    print("🚧🌱🚧 Under Construction 🚧🌱🚧")
