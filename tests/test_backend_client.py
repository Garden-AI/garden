# flake8: noqa: F841
import pytest

import boto3

from garden_ai import Garden, Entrypoint
from garden_ai.gardens import GardenMetadata
from garden_ai.entrypoints import RegisteredEntrypointMetadata


def test_mint_doi_on_datacite_raises_on_bad_response(
    mocker,
    backend_client,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._post",
        return_value={},
    )

    with pytest.raises(Exception):
        backend_client.mint_doi_on_datacite({})


def test_mint_doi_on_datacite_returns_doi_on_success(
    mocker,
    backend_client,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._post", return_value={"doi": "some/doi"}
    )

    doi = backend_client.mint_doi_on_datacite({})
    assert doi == "some/doi"


def test_update_doi_on_datacite_sends_request_to_backend(
    mocker,
    backend_client,
):

    mock_put = mocker.patch(
        "garden_ai.backend_client.BackendClient._put",
    )

    backend_client.update_doi_on_datacite({})
    mock_put.assert_called()


def test_upload_notebook_raises_on_failure(
    mocker,
    backend_client,
    faker,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._put",
        side_effect=Exception("Intentional Error for Testing."),
    )

    with pytest.raises(Exception):
        backend_client.upload_notebook(
            {}, faker.user_name(), faker.first_name() + ".ipynb"
        )


def test_upload_notebook_returns_notebook_url_on_success(
    mocker,
    faker,
    backend_client,
):
    fake_url = faker.url()
    mocker.patch(
        "garden_ai.backend_client.BackendClient._post",
        return_value={"notebook_url": fake_url},
    )

    url = backend_client.upload_notebook(
        {}, faker.user_name(), faker.first_name() + ".ipynb"
    )

    assert url == fake_url


def test_get_docker_push_session_raises_on_bad_response(
    mocker,
    backend_client,
):

    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value={"bad": "data"},
    )

    with pytest.raises(Exception):
        backend_client.get_docker_push_session()


def test_get_docker_push_session_returns_boto3_session(
    mocker,
    backend_client,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value={
            "AccessKeyId": "string",
            "SecretAccessKey": "string",
            "SessionToken": "string",
            "ECRRepo": "string",
            "RegionName": "us-east-1",
        },
    )

    mock_boto3_session = mocker.MagicMock(spec=boto3.Session)

    mocker.patch(
        "garden_ai.backend_client.boto3.Session",
        return_value=mock_boto3_session,
    )

    session = backend_client.get_docker_push_session()
    assert session == mock_boto3_session


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


def test_put_garden_raises_on_backend_failure(
    mocker,
    backend_client,
    mock_GardenMetadata,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._put",
        side_effect=Exception("Intentional Error for Testing"),
    )

    with pytest.raises(Exception):
        backend_client.put_garden(mock_GardenMetadata)


def test_put_garden_returns_garden_on_success(
    mocker,
    backend_client,
    mock_GardenMetadata,
    garden_nested_metadata_json,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._put",
        return_value=garden_nested_metadata_json,
    )

    updated_garden = backend_client.put_garden(mock_GardenMetadata)

    assert isinstance(updated_garden, Garden)


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


def test_get_entrypoint_meta_data_returns_metadata(
    mocker,
    backend_client,
    entrypoint_metadata_json,
):

    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value=entrypoint_metadata_json,
    )

    entrypoint_meta = backend_client.get_entrypoint_metadata("some/doi")

    assert isinstance(entrypoint_meta, RegisteredEntrypointMetadata)


def test_get_entrypoint_metadata_raises_on_backend_failure(
    mocker,
    backend_client,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        side_effect=Exception("Intentional Error for Testing"),
    )

    with pytest.raises(Exception):
        entrypoint_meta = backend_client.get_entrypoint_metadata("some/doi")


def test_get_entrypoint_returns_entrypoint(
    mocker,
    backend_client,
    entrypoint_metadata_json,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value=entrypoint_metadata_json,
    )

    entrypoint = backend_client.get_entrypoint("some/doi")
    assert isinstance(entrypoint, Entrypoint)


def test_get_entrypoint_raises_on_backend_failure(
    mocker,
    backend_client,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        side_effect=Exception("Intentional Error for Testing"),
    )

    with pytest.raises(Exception):
        entrypoint = backend_client.get_entrypoint("some/doi")


def test_delete_entrypoint_raises_on_backend_failure(
    mocker,
    backend_client,
):

    mocker.patch(
        "garden_ai.backend_client.BackendClient._delete",
        side_effect=Exception("Intentional Error for Testing"),
    )

    with pytest.raises(Exception):
        backend_client.delete_entrypoint("some/doi")


def test_get_entrypoints_raises_on_backend_failure(
    mocker,
    backend_client,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        side_effect=Exception("Intentional Error for Testing"),
    )

    with pytest.raises(Exception):
        backend_client.get_entrypoints(
            dois=["some/doi"],
            tags=["tag1", "tag2"],
            authors=["author1", "author2"],
            draft=True,
            year="2023",
            owner_uuid="someuuid",
            limit=5,
        )


def test_get_entrypoints_returns_list_of_entrypoints(
    mocker,
    backend_client,
    entrypoint_metadata_json,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value=[entrypoint_metadata_json for _ in range(5)],
    )

    entrypoints = backend_client.get_entrypoints(
        # args don't matter for this test
        dois=["some/doi"],
    )

    for ep in entrypoints:
        assert isinstance(ep, Entrypoint)


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


def test_get_user_info_raises_on_backend_failure(
    mocker,
    backend_client,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        side_effect=Exception("Intentional Error for Testing"),
    )

    with pytest.raises(Exception):
        backend_client.get_user_info()


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


def test_put_entrypoint_metadata_raises_on_backend_failure(
    mocker,
    backend_client,
    mock_RegisteredEntrypointMetadata,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._put",
        side_effect=Exception("Intentional Error for Testing"),
    )

    with pytest.raises(Exception):
        backend_client.put_entrypoint_metadata(mock_RegisteredEntrypointMetadata)


def test_put_entrypoint_metadata_returns_entrypoint_metadata_on_success(
    mocker,
    backend_client,
    entrypoint_metadata_json,
    mock_RegisteredEntrypointMetadata,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._put",
        return_value=entrypoint_metadata_json,
    )

    entrypoint_meta = backend_client.put_entrypoint_metadata(
        mock_RegisteredEntrypointMetadata
    )

    assert isinstance(entrypoint_meta, RegisteredEntrypointMetadata)
