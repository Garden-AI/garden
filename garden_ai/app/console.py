from rich.console import Console
from rich.table import Table

from typing import List, Optional, Any

console = Console()


def format_doi_status(doi_status: bool) -> str:
    return "draft" if doi_status else "registered"


# Assuming the boolean value is stored in a field named 'doi_status' in the resource objects
# Update the _get_rich_resource_table function to include the new column and format the doi_status
def _get_rich_resource_table(
    resource_objs: Optional[List[Any]], resource_table_cols: List[str], table_name: str
) -> Table:
    table = Table(title=table_name)

    # Add all the existing columns
    for col in resource_table_cols:
        table.add_column(col)

    # Add the new column for DOI status
    table.add_column("DOI Status")

    if resource_objs:
        for resource_obj in resource_objs:
            row = []
            for field in resource_table_cols:
                row.append(str(getattr(resource_obj, field)))
            # Format the DOI status and add it to the row
            doi_status_formatted = format_doi_status(
                getattr(resource_obj, "doi_status", False)
            )
            row.append(doi_status_formatted)
            table.add_row(*row)

    return table


def _get_rich_resource_table(
    resource_objs: Optional[List[Any]], resource_table_cols: List[str], table_name: str
) -> Table:
    table = Table(title=table_name)

    for col in resource_table_cols:
        if col == "doi_is_draft":
            table.add_column("doi status")
        else:
            table.add_column(col)

    if resource_objs:
        for resource_obj in resource_objs:
            row = []
            for field in resource_table_cols:
                cell_value = getattr(resource_obj, field)
                if field == "doi_is_draft":
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
