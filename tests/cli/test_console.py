import pytest
from rich.table import Table
from typer.testing import CliRunner

from garden_ai.app.main import app
from garden_ai import local_data
from garden_ai.app.console import (
    get_local_garden_rich_table,
    get_local_pipeline_rich_table,
    get_local_model_rich_table,
)

runner = CliRunner()


def test_get_local_garden_table(mocker, garden_client, garden_all_fields, tmp_path):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=tmp_path)
    local_data.put_local_garden(garden_all_fields)

    garden_fields = ["uuid", "doi", "title"]
    garden_table_name = "Local Gardens"
    garden_rows = [
        (str(garden_all_fields.uuid), garden_all_fields.doi, garden_all_fields.title)
    ]
    table = Table(title=garden_table_name)

    for col in garden_fields:
        table.add_column(col)
    for row in garden_rows:
        table.add_row(*(row))

    test_table = get_local_garden_rich_table(
        fields=garden_fields[1:], table_name=garden_table_name
    )
    assert table.__dict__ == test_table.__dict__


def test_get_local_pipeline_table(
    mocker, garden_client, registered_pipeline_toy_example, tmp_path
):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=tmp_path)
    local_data.put_local_pipeline(registered_pipeline_toy_example)

    pipeline_fields = ["uuid", "doi", "title"]
    pipeline_table_name = "Local Pipelines"
    pipeline_rows = [
        (
            str(registered_pipeline_toy_example.uuid),
            registered_pipeline_toy_example.doi,
            registered_pipeline_toy_example.title,
        )
    ]
    table = Table(title=pipeline_table_name)

    for col in pipeline_fields:
        table.add_column(col)
    for row in pipeline_rows:
        table.add_row(*(row))

    test_table = get_local_pipeline_rich_table(
        fields=pipeline_fields[1:], table_name=pipeline_table_name
    )
    assert table.__dict__ == test_table.__dict__


def test_get_local_model_table(mocker, database_with_model, second_draft_of_model):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=database_with_model)
    local_data.put_local_model(second_draft_of_model)

    model_fields = ["model_uri", "model_name", "flavor"]
    model_table_name = "Local Models"
    model_rows = [
        (
            second_draft_of_model.model_uri,
            second_draft_of_model.model_name,
            second_draft_of_model.flavor,
        )
    ]
    model_rows = [
        (
            str(second_draft_of_model.model_uri),
            second_draft_of_model.model_name,
            second_draft_of_model.flavor,
        )
    ]
    table = Table(title=model_table_name)

    for col in model_fields:
        table.add_column(col)
    for row in model_rows:
        table.add_row(*(row))

    test_table = get_local_model_rich_table(
        fields=model_fields[1:], table_name=model_table_name
    )
    assert table.__dict__ == test_table.__dict__
