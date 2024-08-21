import pytest

from garden_ai import Garden
from garden_ai.gardens import GardenMetadata
from garden_ai.entrypoints import Entrypoint, RegisteredEntrypointMetadata


def test_garden_init_raises_if_metadata_entrypoints_dont_match(
    garden_nested_metadata_json,
):
    entrypoints = []
    garden_meta = GardenMetadata(**garden_nested_metadata_json)

    with pytest.raises(ValueError):
        # Giving an empty list of entrypoing should cause an error
        # garden_meta has an entrypoint_id so there is a mismatch
        garden = Garden(garden_meta, entrypoints)  # noqa: F841


def test_garden_init(
    garden_nested_metadata_json,
    entrypoint_metadata_json,
):
    garden_meta = GardenMetadata(**garden_nested_metadata_json)
    entrypoint_meta = RegisteredEntrypointMetadata(**entrypoint_metadata_json)
    entrypoint = Entrypoint(entrypoint_meta)

    garden = Garden(garden_meta, [entrypoint])
    assert isinstance(garden, Garden)
    assert garden.metadata == garden_meta
    assert garden.entrypoints == [entrypoint]


def test_can_call_entrypoints_like_methods(
    garden_nested_metadata_json,
    entrypoint_metadata_json,
    mocker,
):
    garden_meta = GardenMetadata(**garden_nested_metadata_json)
    entrypoint_meta = RegisteredEntrypointMetadata(**entrypoint_metadata_json)
    entrypoint = Entrypoint(entrypoint_meta)

    mock_call = mocker.patch.object(Entrypoint, "__call__")

    garden = Garden(garden_meta, [entrypoint])
    # Call the entrypoint like a method
    garden.predict_defect_level_energies()
    mock_call.assert_called()

    with pytest.raises(AttributeError):
        # but calling some other attribute should still fail
        garden.some_entrypoint_that_does_not_exist()


def test_repr_html_contains_garden_doi_and_entrypoint_dois(
    garden_nested_metadata_json,
    entrypoint_metadata_json,
):
    garden_meta = GardenMetadata(**garden_nested_metadata_json)
    entrypoint_meta = RegisteredEntrypointMetadata(**entrypoint_metadata_json)
    entrypoint = Entrypoint(entrypoint_meta)

    garden = Garden(garden_meta, [entrypoint])
    html = garden._repr_html_()

    assert garden.metadata.doi in html
    for ep in garden.entrypoints:
        assert ep.metadata.doi in html


def test_entrypoint_names_in_dir(
    garden_nested_metadata_json,
    entrypoint_metadata_json,
):
    garden_meta = GardenMetadata(**garden_nested_metadata_json)
    entrypoint_meta = RegisteredEntrypointMetadata(**entrypoint_metadata_json)
    entrypoint = Entrypoint(entrypoint_meta)

    garden = Garden(garden_meta, [entrypoint])

    attrs = dir(garden)

    for ep in garden.entrypoints:
        assert ep.metadata.short_name in attrs