from rich.console import Console
from rich.table import Table

from typing import List, Optional, Any

console = Console()

DOI_STATUS_COLUMN = "doi_is_draft"


def _get_rich_resource_table(
    resource_objs: Optional[List[Any]], resource_table_cols: List[str], table_name: str
) -> Table:
    table = Table(title=table_name)

    for col in resource_table_cols:
        if col == DOI_STATUS_COLUMN:
            table.add_column("doi status")
        else:
            table.add_column(col)

    if resource_objs:
        for resource_obj in resource_objs:
            row = []
            for field in resource_table_cols:
                cell_value = getattr(resource_obj, field)
                if field == DOI_STATUS_COLUMN:
                    cell_value = "draft" if cell_value else "registered"
                row.append(str(cell_value))
            table.add_row(*row)

    return table


def get_local_garden_rich_table(
    resource_table_cols: List[str], table_name: str
) -> Table:
    """Helper: fetch all gardens from ~/.garden/data.json and returns a rich table to print.

    Parameters
    ----------
    resource_table_cols List[str]
        A list of the garden fields you want included as col headers in the table.
    table_name str
        Name of the rich table

    Returns
    -------
    Table
        Returns a rich table of all local gardens.
    """
    from garden_ai.local_data import get_all_local_gardens

    local_gardens = get_all_local_gardens()
    return _get_rich_resource_table(
        resource_objs=local_gardens,
        resource_table_cols=resource_table_cols,
        table_name=table_name,
    )


def get_local_entrypoint_rich_table(
    resource_table_cols: List[str], table_name: str
) -> Table:
    """Helper: fetch all entrypoints from ~/.garden/data.json and returns a rich table to print.

    Parameters
    ----------
    resource_table_cols List[str]
        A list of the entrypoint fields you want included as col headers in the table.
    table_name str
        Name of the rich table

    Returns
    -------
    Table
        Returns a rich table of all local entrypoints.
    """
    from garden_ai.local_data import get_all_local_entrypoints

    local_entrypoints = get_all_local_entrypoints()
    return _get_rich_resource_table(
        resource_objs=local_entrypoints,
        resource_table_cols=resource_table_cols,
        table_name=table_name,
    )


def print_err(message: str):
    console.print(message, style="bold red")
