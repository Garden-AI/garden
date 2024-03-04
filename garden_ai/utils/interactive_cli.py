from typing import Union, List, TypeVar

import typer
from prompt_toolkit.shortcuts import radiolist_dialog
from prompt_toolkit import prompt

from garden_ai import RegisteredEntrypoint, Garden

T = TypeVar("T", RegisteredEntrypoint, Garden)


def gui_edit_garden_entity(
    entity: T, string_fields: List[str], list_fields: List[str]
) -> T:
    choices = [
        (field, f"{field}: {getattr(entity, field)}")
        for field in string_fields + list_fields
    ]
    selected_field = radiolist_dialog(
        title="Select Field to Edit",
        text="Use arrow keys to move, enter to select",
        values=choices,
    ).run()

    if not selected_field:
        return

    if selected_field in string_fields:
        old_value = getattr(entity, selected_field)
        new_value = prompt(f'Edit "{selected_field}"\n\n', default=old_value)

    if selected_field in list_fields:
        old_value = getattr(entity, selected_field)
        as_string = ", ".join(old_value)
        new_value_as_string = prompt(
            f'Edit list "{selected_field}". Values are comma-separated.\n\n',
            default=as_string,
        )
        new_value = [x.strip() for x in new_value_as_string.split(",")]

    typer.confirm(
        f'You want to update "{selected_field}" to "{new_value}". Does this look right?',
        abort=True,
    )
    setattr(entity, selected_field, new_value)
