from typing import List

from garden_ai import local_data
from garden_ai.mlmodel import (
    DatasetConnection,
    ModelNotFoundException,
)

from garden_ai.app.console import console, get_local_model_rich_table
from garden_ai.app.completion import complete_model
import typer
import rich
from rich.prompt import Prompt
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
def add_dataset(
    model_name: str = typer.Option(
        ...,
        "-m",
        "--model",
        autocompletion=complete_model,
        help="The name of the model you would like to link your dataset to",
        rich_help_panel="Required",
    ),
    title: str = typer.Option(
        ...,
        "-t",
        "--title",
        prompt="Please enter a short and descriptive title of your dataset",
        rich_help_panel="Required",
    ),
    url: str = typer.Option(
        ...,
        "-u",
        "--url",
        prompt="If you trained this model on a dataset, include a link to the dataset. e.g., Kaggle, Foundry, Zenodo...",
        rich_help_panel="Required",
    ),
    doi: str = typer.Option(
        None,
        "-d",
        "--doi",
        help="If dataset has a DOI it can be referenced by, include the DOI.",
        rich_help_panel="Recommended",
    ),
    data_type: str = typer.Option(
        None,
        "-da",
        "--datatype",
        help="Please enter the file type of data in this dataset. e.g.: .csv,.json,.hdf5,...",
        rich_help_panel="Recommended",
    ),
):
    model = local_data.get_local_model_by_name(model_name)
    if not model:
        raise ModelNotFoundException("This model cannot be found.")
    if not doi:
        doi = Prompt.ask("Add the DOI of the dataset? (leave blank to skip)")
    if not data_type:
        data_type = Prompt.ask("Add the filetype of the dataset (leave blank to skip)")
    only_one_dataset_option_provided = (url and not doi) or (doi and not url)
    if only_one_dataset_option_provided:
        logger.warning(
            "If you are linking a Foundry dataset, please include both --url and --doi"
        )
    if not url and not doi and not data_type:
        raise typer.BadParameter(
            "The parameters of your dataset are empty. Please input more information"
            "about the dataset you would like to link with your model."
        )
    local_dataset = DatasetConnection(
        title=title, doi=doi, data_type=data_type, url=url
    )
    model.dataset = local_dataset
    local_data.put_local_model(model)
    rich.print(f"Successfully added dataset to model {model_name}")


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
        help="The full model names of the models you want to show the local data for. "
        "e.g. ``model show email@addr.ess-model-name/2 email@addr.ess-model-name-2/4`` will show the local data for both models listed.",
        autocompletion=complete_model,
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
