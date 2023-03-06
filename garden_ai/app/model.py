from pathlib import Path
from typing import Optional, List

from garden_ai.client import GardenClient

import typer
import rich

model_app = typer.Typer(name="model", no_args_is_help=True)


@model_app.callback()
def model():
    """
    sub-commands for managing machine learning models
    """
    pass


@model_app.command(no_args_is_help=True)
def register(
    name: str = typer.Argument(
        ...,
        help=("The name of your model"),
    ),
    model_path: Path = typer.Argument(
        ...,
        dir_okay=False,
        file_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
        help=("The path to your model on your filesystem"),
    ),
    flavor: str = typer.Argument(
        "sklearn",
        help=(
            "What ML library did you make the model with? "
            "Currently we just support sklearn."
        ),
    ),
    extra_pip_requirements: Optional[List[str]] = typer.Option(
        None,
        "--extra-pip-requirements",
        "-r",
        help=(
            "Additonal package requirmeents. Add multiple like "
            '--extra_pip_requirements "torch=1.3.1" --extra_pip_requirements "pandas<=1.5.0"'
        ),
    ),
):
    """Register a model in Garden. Outputs a full model identifier that you can reference in a Pipeline."""
    if flavor != "sklearn":
        rich.print("Sorry jk, we only support sklearn.")

    client = GardenClient()
    full_model_name = client.log_model(str(model_path), name, extra_pip_requirements)
    rich.print(
        f"Successfully uploaded your model! The full name to include in your pipeline is '{full_model_name}'"
    )
