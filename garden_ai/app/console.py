from rich.console import Console
from rich.table import Table

from typing import List, Optional, Any

console = Console()


def _get_rich_resource_table(
    resource_objs: Optional[List[Any]], resource_table_cols: List[str], table_name: str
) -> Table:
    table = Table(title=table_name)

    for col in resource_table_cols:
        table.add_column(col)

    if resource_objs:
        for resource_obj in resource_objs:
            row = []
            for field in resource_table_cols:
                row.append(str(getattr(resource_obj, field)))
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


def get_local_pipeline_rich_table(
    resource_table_cols: List[str], table_name: str
) -> Table:
    """Helper: fetch all pipelines from ~/.garden/data.json and returns a rich table to print.

    Parameters
    ----------
    resource_table_cols List[str]
        A list of the pipeline fields you want included as col headers in the table.
    table_name str
        Name of the rich table

    Returns
    -------
    Table
        Returns a rich table of all local pipelines.
    """
    from garden_ai.local_data import get_all_local_pipelines

    local_pipelines = get_all_local_pipelines()
    return _get_rich_resource_table(
        resource_objs=local_pipelines,
        resource_table_cols=resource_table_cols,
        table_name=table_name,
    )


def get_local_model_rich_table(
    resource_table_cols: List[str], table_name: str
) -> Table:
    """Helper: fetch all models from ~/.garden/data.json and returns a rich table to print.

    Parameters
    ----------
    resource_table_cols List[str]
        A list of the model fields you want included as col headers in the table.
    table_name str
        Name of the rich table

    Returns
    -------
    Table
        Returns a rich table of all local models.
    """
    from garden_ai.local_data import get_all_local_models

    local_models = get_all_local_models()
    return _get_rich_resource_table(
        resource_objs=local_models,
        resource_table_cols=resource_table_cols,
        table_name=table_name,
    )
