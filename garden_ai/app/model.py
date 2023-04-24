from pathlib import Path
from typing import Optional, List

from garden_ai.client import GardenClient
from garden_ai.mlmodel import RegisteredModel, LocalModel

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
        exists=True,
        dir_okay=True,
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
            "Currently we support the following flavors 'sklearn', 'tensorflow', and 'pytorch'."
        ),
    ),
    extra_pip_requirements: Optional[List[str]] = typer.Option(
        None,
        "--extra-pip-requirements",
        "-r",
        help=(
            "Additonal package requirmeents. Add multiple like "
            '--extra-pip-requirements "torch=1.3.1" --extra-pip-requirements "pandas<=1.5.0"'
        ),
    ),
):
    """Register a model in Garden. Outputs a full model identifier that you can reference in a Pipeline."""
    if flavor not in ["sklearn", "tensorflow", "pytorch"]:
        raise typer.BadParameter(
            f"Sorry, we only support 'sklearn', 'tensorflow', and 'pytorch'. The {flavor} flavor is not yet supported."
        )
    client = GardenClient()
    local_model = LocalModel(
        model_path=str(model_path),
        name=name,
        flavor=flavor,
        extra_pip_requirements=extra_pip_requirements,
        user_email=client.get_email(),
    )
    registered_model = client.register_model(local_model)
    model_uri = registered_model.model_uri
    rich.print(
        f"Successfully uploaded your model! The full name to include in your pipeline is '{model_uri}'"
    )
