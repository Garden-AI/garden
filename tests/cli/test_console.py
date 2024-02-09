from rich.table import Table

from garden_ai import local_data
from garden_ai.app.console import (
    get_local_garden_rich_table,
    get_local_entrypoint_rich_table,
    DOI_STATUS_COLUMN,
)


def test_get_local_garden_table(mocker, garden_all_fields, tmp_path):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=tmp_path)
    local_data.put_local_garden(garden_all_fields)

    # Construct a table manually
    garden_col_literal_namess = ["doi", "title", "description", "doi status"]
    garden_table_name = "Local Gardens"
    garden_rows = [
        (
            garden_all_fields.doi,
            garden_all_fields.title,
            garden_all_fields.description,
            "draft",
        )
    ]
    table = Table(title=garden_table_name)

    for col in garden_col_literal_namess:
        table.add_column(col)
    for row in garden_rows:
        table.add_row(*(row))

    # Construct it using the helper under test
    garden_col_names = ["doi", "title", "description", DOI_STATUS_COLUMN]
    test_table = get_local_garden_rich_table(
        resource_table_cols=garden_col_names, table_name=garden_table_name
    )
    assert table.__dict__ == test_table.__dict__


def test_get_local_entrypoint_table(
    mocker, registered_entrypoint_toy_example, tmp_path
):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=tmp_path)
    local_data.put_local_entrypoint(registered_entrypoint_toy_example)

    # Construct a table manually
    entrypoint_col_literal_names = ["doi", "title", "description", "doi status"]
    entrypoint_table_name = "Local Entrypoints"
    entrypoint_rows = [
        (
            registered_entrypoint_toy_example.doi,
            registered_entrypoint_toy_example.title,
            registered_entrypoint_toy_example.description,
            "draft",
        )
    ]
    table = Table(title=entrypoint_table_name)

    for col in entrypoint_col_literal_names:
        table.add_column(col)
    for row in entrypoint_rows:
        table.add_row(*(row))

    # Construct it using the helper under test
    entrypoint_col_names = ["doi", "title", "description", DOI_STATUS_COLUMN]
    test_table = get_local_entrypoint_rich_table(
        resource_table_cols=entrypoint_col_names, table_name=entrypoint_table_name
    )
    assert table.__dict__ == test_table.__dict__
