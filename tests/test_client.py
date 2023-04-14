import pytest
import json

from garden_ai import GardenClient
from garden_ai import local_data
from garden_ai.client import AuthException
from globus_sdk import AuthAPIError, AuthClient, OAuthTokenResponse, SearchClient


def test_client_no_previous_tokens(
    mocker, mock_authorizer_tuple, token, mock_keystore, identity_jwt
):
    mock_authorizer_constructor, mock_authorizer = mock_authorizer_tuple
    # Mocks for KeyStore
    mock_keystore.file_exists.return_value = False

    # Mocks for Login Flow
    mock_auth_client = mocker.MagicMock(AuthClient)
    mock_auth_client.oauth2_get_authorize_url = mocker.Mock(
        return_value="https://globus.auth.garden"
    )
    mock_auth_client.oauth2_start_flow = mocker.Mock()
    mocker.patch("garden_ai.client.Prompt.ask").return_value = "my token"
    mocker.patch("garden_ai.client.typer.launch")

    mock_search_client = mocker.MagicMock(SearchClient)

    mock_token_response = mocker.MagicMock(OAuthTokenResponse)
    mock_token_response.by_resource_server = {
        "groups.api.globus.org": token,
        "search.api.globus.org": token,
        "0948a6b0-a622-4078-b0a4-bfd6d77d65cf": token,
        "funcx_service": token,
        "auth.globus.org": token,
    }
    mock_token_response.data = {"id_token": identity_jwt}
    mock_auth_client.oauth2_exchange_code_for_tokens = mocker.Mock(
        return_value=mock_token_response
    )

    # Call the Garden constructor
    gc = GardenClient(auth_client=mock_auth_client, search_client=mock_search_client)

    assert gc.auth_key_store == mock_keystore
    mock_auth_client.oauth2_exchange_code_for_tokens.assert_called_with("my token")

    mock_auth_client.oauth2_start_flow.assert_called_with(
        refresh_tokens=True,
        requested_scopes=[
            "openid",
            "email",
            "urn:globus:auth:scope:groups.api.globus.org:view_my_groups_and_memberships",
            "urn:globus:auth:scope:search.api.globus.org:ingest",
            "https://auth.globus.org/scopes/0948a6b0-a622-4078-b0a4-bfd6d77d65cf/action_all",
            "https://auth.globus.org/scopes/facd7ccc-c5f4-42aa-916b-a0e270e2c2a9/all",
        ],
    )

    mock_keystore.store.assert_called_with(mock_token_response)
    mock_authorizer_constructor.assert_called_with(
        "MyRefreshToken",
        mock_auth_client,
        access_token="MyAccessToken",
        expires_at=1024,
        on_refresh=mock_keystore.on_refresh,
    )

    assert gc.groups_authorizer == mock_authorizer


def test_client_previous_tokens_stored(
    mocker, mock_authorizer_tuple, token, mock_keystore
):
    mock_authorizer_constructor, mock_authorizer = mock_authorizer_tuple
    mock_auth_client = mocker.MagicMock(AuthClient)

    mock_keystore.file_exists.return_value = True
    mock_keystore.get_token_data.return_value = token

    # Call the Garden constructor
    gc = GardenClient(auth_client=mock_auth_client)

    assert gc.auth_key_store == mock_keystore
    mock_auth_client.oauth2_exchange_code_for_tokens.assert_not_called()

    mock_authorizer_constructor.assert_called_with(
        "MyRefreshToken",
        mock_auth_client,
        access_token="MyAccessToken",
        expires_at=1024,
        on_refresh=mock_keystore.on_refresh,
    )

    assert gc.groups_authorizer == mock_authorizer


def test_client_invalid_auth_token(
    mocker, mock_authorizer_tuple, token, mock_keystore, identity_jwt
):
    mock_authorizer_constructor, mock_authorizer = mock_authorizer_tuple
    # Mocks for KeyStore
    mock_keystore.file_exists.return_value = False

    # Mocks for Login Flow
    mock_auth_client = mocker.MagicMock(AuthClient)
    mock_auth_client.oauth2_get_authorize_url = mocker.Mock(
        return_value="https://globus.auth.garden"
    )
    mock_auth_client.oauth2_start_flow = mocker.Mock()
    mocker.patch("garden_ai.client.Prompt.ask").return_value = "my token"
    mocker.patch("garden_ai.client.typer.launch")

    mock_token_response = mocker.MagicMock(OAuthTokenResponse)
    mock_token_response.data = {"id_token": identity_jwt}
    mock_token_response.by_resource_server = {"groups.api.globus.org": token}
    mock_token_response.status_code = "X"
    mock_token_response.request = mocker.Mock()
    mock_token_response.request.headers = {"Authorization": "Yougotit"}
    mock_token_response.request._underlying_response = mocker.Mock()
    mock_token_response.url = "http://foo.bar.baz"
    mock_auth_client.oauth2_exchange_code_for_tokens.side_effect = AuthAPIError(
        r=mock_token_response
    )
    # Call the Garden constructor and expect an auth exception
    with pytest.raises(AuthException):
        GardenClient(auth_client=mock_auth_client)


# TODO: carve out local storage tests
def test_local_storage_garden(mocker, garden_client, garden_all_fields, tmp_path):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.client.LOCAL_STORAGE", new=tmp_path)
    uuid = garden_all_fields.uuid
    local_data.put_local_garden(garden_all_fields)
    record = local_data.get_local_garden_by_uuid(uuid)
    assert json.dumps(record) == garden_all_fields.json()


def test_local_storage_pipeline(mocker, garden_client, pipeline_toy_example, tmp_path):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.client.LOCAL_STORAGE", new=tmp_path)
    uuid = pipeline_toy_example.uuid
    local_data.put_local_pipeline(pipeline_toy_example)
    record = local_data.get_local_pipeline_by_uuid(uuid)
    assert json.dumps(record) == pipeline_toy_example.json()


def test_local_storage_keyerror(mocker, garden_client, garden_all_fields, tmp_path):
    # mock to replace "~/.garden/db"
    tmp_path.mkdir(parents=True, exist_ok=True)
    mocker.patch("garden_ai.client.LOCAL_STORAGE", new=tmp_path)
    pipeline, *_ = garden_all_fields.pipelines
    # put the pipeline, not garden (hence db is nonempty)
    local_data.put_local_pipeline(pipeline)

    # can't find the garden
    assert local_data.get_local_garden_by_uuid(garden_all_fields.uuid) is None

    # can find the pipeline
    record = local_data.get_local_pipeline_by_uuid(pipeline.uuid)
    assert json.dumps(record) == pipeline.json()
