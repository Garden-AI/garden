import pytest

from garden_ai import GardenClient
from globus_sdk import AuthClient, OAuthTokenResponse
from globus_sdk.tokenstorage import SimpleJSONFileAdapter


@pytest.fixture
def mock_authorizer_tuple(mocker):
    mock_authorizer = mocker.Mock()
    mock_authorizer_constructor = mocker.patch("garden_ai.garden.RefreshTokenAuthorizer",
                                               return_value=mock_authorizer)
    return mock_authorizer_constructor, mock_authorizer


@pytest.fixture
def token():
    return {
        "refresh_token": "MyRefreshToken",
        "access_token": "MyAccessToken",
        "expires_at_seconds": 1024
    }


@pytest.fixture
def mock_keystore(mocker):
    mock_keystore = mocker.MagicMock(SimpleJSONFileAdapter)
    mocker.patch("garden_ai.garden.SimpleJSONFileAdapter").return_value = mock_keystore
    return mock_keystore


def test_client_no_previous_tokens(mocker, mock_authorizer_tuple, token, mock_keystore):
    mock_authorizer_constructor, mock_authorizer = mock_authorizer_tuple
    # Mocks for KeyStore
    mock_keystore.file_exists.return_value = False

    # Mocks for Login Flow
    mock_auth_client = mocker.MagicMock(AuthClient)
    mock_auth_client.oauth2_get_authorize_url = mocker.Mock(
        return_value="https://globus.auth.garden")
    mock_auth_client.oauth2_start_flow = mocker.Mock()
    mocker.patch("garden_ai.garden.input").return_value = "my token"

    mock_token_response = mocker.MagicMock(OAuthTokenResponse)
    mock_token_response.by_resource_server = {"groups.api.globus.org": token}
    mock_auth_client.oauth2_exchange_code_for_tokens = mocker.Mock(
        return_value=mock_token_response)

    # Call the Garden constructor
    gc = GardenClient(auth_client=mock_auth_client)

    assert gc.auth_key_store == mock_keystore
    mock_auth_client.oauth2_exchange_code_for_tokens.assert_called_with("my token")

    mock_auth_client.oauth2_start_flow.assert_called_with(
        refresh_tokens=True,
        requested_scopes="urn:globus:auth:scope:groups.api.globus.org:view_my_groups_and_memberships")

    mock_keystore.store.assert_called_with(mock_token_response)
    mock_authorizer_constructor.assert_called_with("MyRefreshToken", mock_auth_client,
                                                   access_token="MyAccessToken",
                                                   expires_at=1024,
                                                   on_refresh=mock_keystore.on_refresh)

    assert gc.authorizer == mock_authorizer


def test_client_previous_tokens_stored(mocker, mock_authorizer_tuple, token, mock_keystore):
    mock_authorizer_constructor, mock_authorizer = mock_authorizer_tuple
    mock_auth_client = mocker.MagicMock(AuthClient)

    mock_keystore.file_exists.return_value = True
    mock_keystore.get_token_data.return_value = token

    # Call the Garden constructor
    gc = GardenClient(auth_client=mock_auth_client)

    assert gc.auth_key_store == mock_keystore
    mock_auth_client.oauth2_exchange_code_for_tokens.assert_not_called()

    mock_authorizer_constructor.assert_called_with("MyRefreshToken", mock_auth_client,
                                                   access_token="MyAccessToken",
                                                   expires_at=1024,
                                                   on_refresh=mock_keystore.on_refresh)

    assert gc.authorizer == mock_authorizer
