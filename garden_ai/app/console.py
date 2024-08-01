from rich.console import Console
from rich.table import Table

from typing import List, Optional, Any
from garden_ai.client import GardenClient

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
                cell_value = getattr(resource_obj.metadata, field)
                if field == DOI_STATUS_COLUMN:
                    cell_value = "draft" if cell_value else "registered"
                row.append(str(cell_value))
            table.add_row(*row)

    return table


def get_owned_gardens_rich_table(
    client: GardenClient, resource_table_cols: list[str], table_name: str
):
    owner_uuid = client.get_user_identity_id()
    owned_gardens = client.backend_client.get_gardens(owner_uuid=owner_uuid)
    return _get_rich_resource_table(
        resource_objs=owned_gardens,
        resource_table_cols=resource_table_cols,
        table_name=table_name,
    )


def get_owned_entrypoints_rich_table(
    client: GardenClient, resource_table_cols: list[str], table_name: str, limit=100
):
    owner_uuid = client.get_user_identity_id()
    owned_entrypoints = client.backend_client.get_entrypoints(
        owner_uuid=owner_uuid, limit=limit
    )
    return _get_rich_resource_table(
        resource_objs=owned_entrypoints,
        resource_table_cols=resource_table_cols,
        table_name=table_name,
    )


def print_err(message: str):
    console.print(message, style="bold red")
