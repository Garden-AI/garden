from copy import deepcopy

import pytest

from garden_ai import Garden
from garden_ai.gardens import GardenMetadata
from garden_ai.modal.classes import ModalClassWrapper
from garden_ai.modal.functions import ModalFunction, ModalFunctionMetadata


def test_garden_init_raises_if_metadata_functions_dont_match(
    garden_nested_metadata_json,
):
    # Giving an empty list of Modal functions should cause an error
    # Because there is at least one modal fn ID in the metadata
    data = deepcopy(garden_nested_metadata_json)
    garden_meta = GardenMetadata(**data)
    modal_functions = []
    with pytest.raises(ValueError):
        garden = Garden(garden_meta, modal_functions)  # noqa: F841


def test_garden_init(
    garden_nested_metadata_json,
    modal_function_metadata_json,
    garden_client,
):
    garden_meta = GardenMetadata(**garden_nested_metadata_json)
    modal_function_meta = ModalFunctionMetadata(**modal_function_metadata_json)
    modal_function = ModalFunction(modal_function_meta, garden_client)

    garden = Garden(garden_meta, [modal_function])
    assert isinstance(garden, Garden)
    assert garden.metadata == garden_meta
    assert garden.modal_functions == [modal_function]


def test_can_call_modal_functions_like_methods(
    garden_nested_metadata_json,
    modal_function_metadata_json,
    garden_client,
    mocker,
):
    data = deepcopy(garden_nested_metadata_json)
    garden_meta = GardenMetadata(**data)
    modal_function_meta = ModalFunctionMetadata(**modal_function_metadata_json)
    modal_function = ModalFunction(modal_function_meta, garden_client)

    mock_call = mocker.patch.object(ModalFunction, "__call__")

    garden = Garden(garden_meta, [modal_function])
    # Call the function like a method
    garden.test_function_name()
    mock_call.assert_called()

    with pytest.raises(AttributeError):
        # but calling some other attribute should still fail
        garden.does_not_exist()


def test_can_call_modal_methods(
    garden_nested_metadata_json_with_modal_class,
    modal_method_metadata_json,
    garden_client,
    mocker,
):
    data = deepcopy(garden_nested_metadata_json_with_modal_class)
    garden_meta = GardenMetadata(**data)
    modal_function_meta = ModalFunctionMetadata(**modal_method_metadata_json)
    modal_function = ModalFunction(modal_function_meta, garden_client)
    modal_class = ModalClassWrapper("ClassName", [modal_function])

    mock_call = mocker.patch.object(ModalFunction, "__call__")

    garden = Garden(garden_meta, [], [modal_class])
    # Call the method on the class wrapper
    garden.ClassName.method_name()
    mock_call.assert_called()


def test_repr_html_contains_garden_doi_and_entrypoint_dois(garden_nested_metadata_json):
    data = deepcopy(garden_nested_metadata_json)
    del data["modal_function_ids"]
    garden_meta = GardenMetadata(**data)

    garden = Garden(garden_meta)
    html = garden._repr_html_()

    assert garden.metadata.doi in html
