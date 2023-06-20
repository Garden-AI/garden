import pytest

import os

from garden_ai import GardenClient
from garden_ai.client import AuthException
from globus_sdk import (
    AuthAPIError,
    AuthClient,
    OAuthTokenResponse,
    SearchClient,
    ClientCredentialsAuthorizer,
    ConfidentialAppAuthClient,
)
from globus_compute_sdk import Client  # type: ignore

is_gha = os.getenv("GITHUB_ACTIONS")


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
            "urn:globus:auth:scope:search.api.globus.org:all",
            "https://auth.globus.org/scopes/0948a6b0-a622-4078-b0a4-bfd6d77d65cf/test_scope",
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


@pytest.mark.skipif((not is_gha), reason="Test only works in Github Actions.")
def test_client_credentials_grant(cc_grant_tuple):
    # Must run as github action to get client id and client secret from env vars
    client_id = cc_grant_tuple[0]
    client_secret = cc_grant_tuple[1]

    confidential_client = ConfidentialAppAuthClient(client_id, client_secret)
    gc = GardenClient(auth_client=confidential_client)

    assert isinstance(gc.openid_authorizer, ClientCredentialsAuthorizer)
    assert isinstance(gc.groups_authorizer, ClientCredentialsAuthorizer)
    assert isinstance(gc.search_authorizer, ClientCredentialsAuthorizer)
    assert isinstance(gc.compute_authorizer, ClientCredentialsAuthorizer)
    assert isinstance(gc.garden_authorizer, ClientCredentialsAuthorizer)

    assert isinstance(gc.compute_client, Client)
    assert isinstance(gc.search_client, SearchClient)

    assert isinstance(gc.auth_client, ConfidentialAppAuthClient)

    assert gc.auth_client.oauth2_validate_token(gc.openid_authorizer.access_token)[
        "active"
    ]
    assert gc.auth_client.oauth2_validate_token(gc.groups_authorizer.access_token)[
        "active"
    ]
    assert gc.auth_client.oauth2_validate_token(gc.search_authorizer.access_token)[
        "active"
    ]
    assert gc.auth_client.oauth2_validate_token(gc.compute_authorizer.access_token)[
        "active"
    ]
    assert gc.auth_client.oauth2_validate_token(gc.garden_authorizer.access_token)[
        "active"
    ]
