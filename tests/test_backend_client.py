# flake8: noqa: F841
import pytest

from garden_ai import Garden
from garden_ai.gardens import GardenMetadata


def test_get_garden_returns_garden(
    mocker,
    backend_client,
    garden_nested_metadata_json,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value=garden_nested_metadata_json,
    )

    garden = backend_client.get_garden("some/doi")
    assert isinstance(garden, Garden)


def test_get_garden_raises_on_backend_failure(
    mocker,
    backend_client,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        side_effect=Exception("Intentional Error for Testing"),
    )

    with pytest.raises(Exception):
        garden = backend_client.get_garden("some/doi")


def test_get_garden_metadata_raises_on_backend_failure(
    mocker,
    backend_client,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        side_effect=Exception("Intentional Error for Testing"),
    )

    with pytest.raises(Exception):
        garden_meta = backend_client.get_garden_metadata("some/doi")


def test_get_garden_metadata_returns_garden_on_success(
    mocker,
    backend_client,
    garden_nested_metadata_json,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value=garden_nested_metadata_json,
    )

    garden_meta = backend_client.get_garden_metadata("some/doi")

    assert isinstance(garden_meta, GardenMetadata)


def test_delete_garden_raises_on_backend_failure(
    mocker,
    backend_client,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._delete",
        side_effect=Exception("Intentional Error for Testing"),
    )

    with pytest.raises(Exception):
        backend_client.delete_garden("some/doi")


def test_get_gardens_raises_on_backend_failure(
    mocker,
    backend_client,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        side_effect=Exception("Intentional Error for Testing"),
    )

    with pytest.raises(Exception):
        backend_client.get_gardens(
            dois=["some/doi"],
            tags=["tag1", "tag2"],
            authors=["author1", "author2"],
            contributors=["contributor1", "contributor2"],
            draft=True,
            year="2023",
            owner_uuid="someuuid",
            limit=5,
        )


def test_get_gardens_returns_list_of_gardens(
    mocker,
    backend_client,
    garden_nested_metadata_json,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value=[garden_nested_metadata_json for _ in range(5)],
    )

    gardens = backend_client.get_gardens(
        # args don't matter for this test
        dois=["some/doi"],
    )

    for garden in gardens:
        assert isinstance(garden, Garden)


def test_get_user_info_returns_user_info(
    mocker,
    backend_client,
    mock_user_info_response,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value=mock_user_info_response,
    )

    user_info = backend_client.get_user_info()
    assert user_info == mock_user_info_response


def test_get_garden_raises_on_archived_garden(
    mocker,
    backend_client,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value={"is_archived": True},
    )

    with pytest.raises(Exception):
        backend_client.get_garden("some/doi")
