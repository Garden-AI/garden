from pathlib import Path
from typing import Optional, List

from garden_ai import local_data
from garden_ai.client import GardenClient
from garden_ai.mlmodel import (
    DatasetConnection,
    LocalModel,
    ModelFlavor,
)
from garden_ai.app.console import console, get_local_model_rich_table

import typer
import rich
import logging

model_app = typer.Typer(name="model", no_args_is_help=True)

logger = logging.getLogger()


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
    dataset_url: Optional[str] = typer.Option(
        None,
        "--dataset-url",
        help=(
            "If you trained this model on a Foundry dataset, include a link to the dataset with this option"
        ),
    ),
    dataset_doi: Optional[str] = typer.Option(
        None,
        "--dataset-doi",
        help=(
            "If you trained this model on a Foundry dataset, include the doi of the dataset"
        ),
    ),
):
    """Register a model in Garden. Outputs a full model identifier that you can reference in a Pipeline."""
    if flavor not in [f.value for f in ModelFlavor]:
        raise typer.BadParameter(
            f"Sorry, we only support 'sklearn', 'tensorflow', and 'pytorch'. The {flavor} flavor is not yet supported."
        )

    only_one_dataset_option_provided = (dataset_url and not dataset_doi) or (
        dataset_doi and not dataset_url
    )
    if only_one_dataset_option_provided:
        raise typer.BadParameter(
            "If you are linking a Foundry dataset, please include both --dataset-url and --dataset-doi"
        )

    client = GardenClient()
    local_model = LocalModel(
        local_path=str(model_path),
        model_name=name,
        flavor=flavor,
        user_email=client.get_email(),
    )
    if dataset_doi and dataset_url:
        dataset_metadata = DatasetConnection(doi=dataset_doi, url=dataset_url)
        local_model.connections.append(dataset_metadata)

    registered_model = client.register_model(local_model)
    rich.print(
        f"Successfully uploaded your model! The full name to include in your pipeline is '{registered_model.full_name}'"
    )


@model_app.command(no_args_is_help=False)
def list():
    """Lists all local models."""

    resource_table_cols = ["full_name", "model_name", "flavor"]
    table_name = "Local Models"

    table = get_local_model_rich_table(
        resource_table_cols=resource_table_cols, table_name=table_name
    )
    console.print("\n")
    console.print(table)


@model_app.command(no_args_is_help=True)
def show(
    model_ids: List[str] = typer.Argument(
        ...,
        help="The URIs of the models you want to show the local data for. "
        "e.g. ``model show email@addr.ess-model-name/2 email@addr.ess-model-name-2/4`` will show the local data for both models listed.",
    ),
):
    """Shows all info for some Models"""

    for model_id in model_ids:
        model = local_data.get_local_model_by_name(model_id)
        if model:
            rich.print(f"Model: {model_id} local data:")
            rich.print_json(json=model.json())
            rich.print("\n")
        else:
            rich.print(f"Could not find model with id {model_id}\n")
