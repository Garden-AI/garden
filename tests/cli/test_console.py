from rich.table import Table

from garden_ai import local_data
from garden_ai.app.console import (
    get_local_garden_rich_table,
    get_local_pipeline_rich_table,
    get_local_model_rich_table,
)


def test_get_local_garden_table(mocker, garden_client, garden_all_fields, tmp_path):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=tmp_path)
    local_data.put_local_garden(garden_all_fields)

    garden_cols = ["doi", "title", "description"]
    garden_table_name = "Local Gardens"
    garden_rows = [
        (garden_all_fields.doi, garden_all_fields.title, garden_all_fields.description)
    ]
    table = Table(title=garden_table_name)

    for col in garden_cols:
        table.add_column(col)
    for row in garden_rows:
        table.add_row(*(row))

    test_table = get_local_garden_rich_table(
        resource_table_cols=garden_cols, table_name=garden_table_name
    )
    assert table.__dict__ == test_table.__dict__


def test_get_local_pipeline_table(
    mocker, garden_client, registered_pipeline_toy_example, tmp_path
):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=tmp_path)
    local_data.put_local_pipeline(registered_pipeline_toy_example)

    pipeline_cols = ["doi", "title", "description"]
    pipeline_table_name = "Local Pipelines"
    pipeline_rows = [
        (
            registered_pipeline_toy_example.doi,
            registered_pipeline_toy_example.title,
            registered_pipeline_toy_example.description,
        )
    ]
    table = Table(title=pipeline_table_name)

    for col in pipeline_cols:
        table.add_column(col)
    for row in pipeline_rows:
        table.add_row(*(row))

    test_table = get_local_pipeline_rich_table(
        resource_table_cols=pipeline_cols, table_name=pipeline_table_name
    )
    assert table.__dict__ == test_table.__dict__


def test_get_local_model_table(mocker, database_with_model, second_draft_of_model):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=database_with_model)
    local_data.put_local_model(second_draft_of_model)

    model_cols = ["model_uri", "model_name", "flavor"]
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

    for col in model_cols:
        table.add_column(col)
    for row in model_rows:
        table.add_row(*(row))

    test_table = get_local_model_rich_table(
        resource_table_cols=model_cols, table_name=model_table_name
    )
    assert table.__dict__ == test_table.__dict__
