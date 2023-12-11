from rich.table import Table

from garden_ai import local_data
from garden_ai.app.console import (
    get_local_garden_rich_table,
    get_local_entrypoint_rich_table,
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


def test_get_local_entrypoint_table(
    mocker, garden_client, registered_entrypoint_toy_example, tmp_path
):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=tmp_path)
    local_data.put_local_entrypoint(registered_entrypoint_toy_example)

    entrypoint_cols = ["doi", "title", "description"]
    entrypoint_table_name = "Local Entrypoints"
    entrypoint_rows = [
        (
            registered_entrypoint_toy_example.doi,
            registered_entrypoint_toy_example.title,
            registered_entrypoint_toy_example.description,
        )
    ]
    table = Table(title=entrypoint_table_name)

    for col in entrypoint_cols:
        table.add_column(col)
    for row in entrypoint_rows:
        table.add_row(*(row))

    test_table = get_local_entrypoint_rich_table(
        resource_table_cols=entrypoint_cols, table_name=entrypoint_table_name
    )
    assert table.__dict__ == test_table.__dict__
