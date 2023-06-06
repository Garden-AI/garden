from rich.console import Console
from rich.table import Table

from typing import List, Any

console = Console()


def _get_rich_resource_table(
    resource_type: Any, fields: List[str], table_name: str
) -> Table:
    from garden_ai import local_data

    table = Table(title=table_name)

    resource_table_cols = [local_data.resource_type_to_id_key[resource_type]]
    resource_table_cols.extend(fields)

    data = local_data._read_local_db()
    resource_data = data.get(resource_type.value)

    if resource_data is not None:
        resource_table_rows = []
        for r_id, r_data in resource_data.items():
            resource_table_row = [r_id]
            for f in fields:
                if r_data[f] is None:
                    resource_table_row.append("None")
                else:
                    resource_table_row.append(r_data[f])
            resource_table_rows.append(tuple(resource_table_row))
        for col in resource_table_cols:
            table.add_column(col)
        for row in resource_table_rows:
            table.add_row(*(row))
    else:
        for col in resource_table_cols:
            table.add_column(col)

    return table


def get_local_garden_rich_table(fields: List[str], table_name: str) -> Table:
    """Helper: fetch all gardens from ~/.garden/data.json and outputs a rich table to print.

    Parameters
    ----------
    fields List
        A list of the garden fields you want included as cols in the table.
    table_name str
        Name of the rich table

    Returns
    -------
    Table
        Returns a rich table of all local gardens.
    """
    from garden_ai.local_data import ResourceType

    return _get_rich_resource_table(
        resource_type=ResourceType.GARDEN, fields=fields, table_name=table_name
    )


def get_local_pipeline_rich_table(fields: List[str], table_name: str) -> Table:
    """Helper: fetch all pipelines from ~/.garden/data.json and outputs a rich table to print.

    Parameters
    ----------
    fields List
        A list of the pipeline fields you want included as cols in the table.
    table_name str
        Name of the rich table

    Returns
    -------
    Table
        Returns a rich table of all local pipelines.
    """
    from garden_ai.local_data import ResourceType

    return _get_rich_resource_table(
        resource_type=ResourceType.PIPELINE, fields=fields, table_name=table_name
    )


def get_local_model_rich_table(fields: List[str], table_name: str) -> Table:
    """Helper: fetch all models from ~/.garden/data.json and outputs a rich table to print.

    Parameters
    ----------
    fields List
        A list of the model fields you want included as cols in the table.
    table_name str
        Name of the rich table

    Returns
    -------
    Table
        Returns a rich table of all local models.
    """
    from garden_ai.local_data import ResourceType

    return _get_rich_resource_table(
        resource_type=ResourceType.MODEL, fields=fields, table_name=table_name
    )
