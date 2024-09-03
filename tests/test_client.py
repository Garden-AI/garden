# flake8: noqa: F841
import os
from urllib.parse import quote

import pytest
from globus_compute_sdk import Client  # type: ignore
from globus_compute_sdk.sdk.login_manager.manager import LoginManager  # type: ignore
from globus_sdk import (
    AuthAPIError,
    AuthLoginClient,
    ClientCredentialsAuthorizer,
    ConfidentialAppAuthClient,
    OAuthTokenResponse,
    SearchClient,
)

from garden_ai import GardenClient, Entrypoint, Garden
from garden_ai.client import AuthException

is_gha = os.getenv("GITHUB_ACTIONS")


def test_client_no_previous_tokens(
    mocker,
    mock_authorizer_tuple,
    token,
    mock_keystore,
    identity_jwt,
):
    mock_authorizer_constructor, mock_authorizer = mock_authorizer_tuple
    # Mocks for KeyStore
    mock_keystore.file_exists.return_value = False

    # Mocks for Login Flow
    mock_auth_client = mocker.MagicMock(AuthLoginClient)
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

    # Mocks compute client login
    mock_login_manager = mocker.MagicMock(LoginManager)
    mock_login_manager.ensure_logged_in = mocker.Mock(return_value=True)
    mocker.patch("globus_compute_sdk.sdk.client.LoginManager").return_value = (
        mock_login_manager
    )

    mocker.patch("garden_ai.client.time.sleep")

    # Call the Garden constructor
    gc = GardenClient(auth_client=mock_auth_client, search_client=mock_search_client)

    assert gc.auth_key_store == mock_keystore
    mock_auth_client.oauth2_exchange_code_for_tokens.assert_called_with("my token")

    mock_auth_client.oauth2_start_flow.assert_called_with(
        refresh_tokens=True,
        requested_scopes=[
            "openid",
            "email",
            "urn:globus:auth:scope:auth.globus.org:manage_projects",
            "urn:globus:auth:scope:groups.api.globus.org:all",
            "urn:globus:auth:scope:search.api.globus.org:all",
            "https://auth.globus.org/scopes/0948a6b0-a622-4078-b0a4-bfd6d77d65cf/test_scope",
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
    mocker,
    mock_authorizer_tuple,
    token,
    mock_keystore,
):
    mock_authorizer_constructor, mock_authorizer = mock_authorizer_tuple
    mock_auth_client = mocker.MagicMock(AuthLoginClient)

    mock_keystore.file_exists.return_value = True
    mock_keystore.get_token_data.return_value = token

    # Mocks compute client login
    mock_login_manager = mocker.MagicMock(LoginManager)
    mock_login_manager.ensure_logged_in = mocker.Mock(return_value=True)
    mocker.patch("globus_compute_sdk.sdk.client.LoginManager").return_value = (
        mock_login_manager
    )

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
    mocker,
    mock_authorizer_tuple,
    token,
    mock_keystore,
    identity_jwt,
):
    mock_authorizer_constructor, mock_authorizer = mock_authorizer_tuple
    # Mocks for KeyStore
    mock_keystore.file_exists.return_value = False

    # Mocks for Login Flow
    mock_auth_client = mocker.MagicMock(AuthLoginClient)
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

    # mock time.sleep so the test runs faster
    mocker.patch("garden_ai.client.time.sleep")

    # Add a json() method to the mock_token_response
    mock_token_response.json = mocker.Mock(return_value={"error": "mock error"})

    # Add the 'reason' attribute to the mock_token_response
    mock_token_response.reason = mocker.Mock(return_value="Mock Reason")

    mock_auth_client.oauth2_exchange_code_for_tokens.side_effect = AuthAPIError(
        r=mock_token_response
    )
    # Call the Garden constructor and expect an auth exception
    with pytest.raises(AuthException):
        GardenClient(auth_client=mock_auth_client)


@pytest.mark.skipif((not is_gha), reason="Test only works in Github Actions.")
@pytest.mark.integration
def test_client_credentials_grant(
    cc_grant_tuple,
):
    # Must run as github action to get client id and client secret from env vars
    client_id = cc_grant_tuple[0]
    client_secret = cc_grant_tuple[1]

    os.environ["FUNCX_SDK_CLIENT_ID"] = client_id
    os.environ["FUNCX_SDK_CLIENT_SECRET"] = client_secret

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


def test_client_datacite_url_correct(
    mocker,
    garden_client,
):

    gc = garden_client

    # Create a mock object for PublishedGarden or RegisteredEntrypoint
    mock_obj = mocker.MagicMock()
    mock_obj.doi = "10.1234/abcd.efgh"  # Set the doi attribute

    # Mock the datacite_json method to return a dummy JSON
    mock_obj._datacite_json.return_value = '{"dummy": "json"}'

    # Mock the BackendClient.update_doi_on_datacite method
    mock_update_doi_on_datacite = mocker.patch(
        "garden_ai.backend_client.BackendClient.update_doi_on_datacite"
    )

    # Call _update_datacite with the mock and tell it to register the doi
    gc._update_datacite(mock_obj, register_doi=True)

    expected_url = f"https://thegardens.ai/#/garden/{mock_obj.doi.replace('/', '%2F')}"

    # Assert that the URL in the payload is correct
    payload = mock_update_doi_on_datacite.call_args[0][0]
    assert payload["data"]["attributes"]["url"] == expected_url


def test_get_entrypoint_get_entrypoint_data_from_backend(
    garden_client,
    mocker,
    mock_RegisteredEntrypointMetadata,
):
    mock_get = mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value=mock_RegisteredEntrypointMetadata.model_dump(mode="json"),
    )

    ep = garden_client.get_entrypoint(mock_RegisteredEntrypointMetadata.doi)

    mock_get.assert_called_once()
    assert ep == Entrypoint(mock_RegisteredEntrypointMetadata)


def test_get_email_returns_correct_email_for_logged_in_user(
    garden_client,
    mocker,
    mock_user_info_response,
):

    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value=mock_user_info_response,
    )

    email = garden_client.get_email()
    assert email == mock_user_info_response["email"]


def test_get_email_raises_if_no_email_or_username(
    garden_client,
    mocker,
    mock_user_info_response,
):
    # Ensure no email or username is present in the response
    del mock_user_info_response["email"]
    del mock_user_info_response["username"]

    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value=mock_user_info_response,
    )

    with pytest.raises(Exception):
        email = garden_client.get_email()


def test_get_email_returns_username_if_no_email(
    garden_client,
    mocker,
    mock_user_info_response,
):
    # Ensure no email is present in the response
    del mock_user_info_response["email"]

    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value=mock_user_info_response,
    )

    username = garden_client.get_email()
    assert username == mock_user_info_response["username"]


def test_get_user_identity_id_returns_correct_id(
    garden_client,
    mocker,
    mock_user_info_response,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value=mock_user_info_response,
    )

    identity_id = garden_client.get_user_identity_id()
    assert identity_id == mock_user_info_response["identity_id"]


def test_upload_notebook_returns_notebook_url(
    mocker,
    faker,
    garden_client,
    mock_user_info_response,
):

    notebook_contents = {}
    notebook_name = faker.first_name() + ".ipynb"

    fake_url = faker.url()
    mocker.patch(
        "garden_ai.backend_client.BackendClient._post",
        return_value={"notebook_url": fake_url},
    )
    mocker.patch(
        "garden_ai.backend_client.BackendClient.get_user_info",
        return_value=mock_user_info_response,
    )

    url = garden_client.upload_notebook(notebook_contents, notebook_name)
    assert url == fake_url


def test_upload_notebook_raises_on_upload_failure(
    garden_client,
    mocker,
    faker,
    mock_user_info_response,
):
    mocker.patch(
        "garden_ai.backend_client.BackendClient.get_user_info",
        return_value=mock_user_info_response,
    )

    # Simulate an error response from the backend
    mocker.patch(
        "garden_ai.backend_client.BackendClient._post",
        side_effect=Exception("Intentional Error for Testing"),
    )

    notebook_contents = {}
    notebook_name = faker.first_name() + ".ipynb"
    with pytest.raises(Exception):
        garden_client.upload_notebook(notebook_contents, notebook_name)


def test_search_forwards_query_to_search_index(
    garden_client,
    mocker,
):
    mock_globus_search = mocker.patch(
        "garden_ai.globus_search.garden_search.search_gardens",
        return_value="I am a search result!",
    )

    fake_query = "I am a search query!"

    result = garden_client.search(fake_query)

    assert fake_query in mock_globus_search.call_args[0]


def test_get_garden_returns_garden_from_backend(
    garden_client,
    mocker,
    garden_nested_metadata_json,
):

    mock_get = mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value=garden_nested_metadata_json,
    )

    garden = garden_client.get_garden(garden_nested_metadata_json["doi"])

    mock_get.assert_called_once()
    assert isinstance(garden, Garden)
    assert garden.metadata.doi == garden_nested_metadata_json["doi"]


def test_create_garden_posts_garden_metadata_to_backend(
    garden_client,
    mocker,
    garden_nested_metadata_json,
    mock_GardenMetadata,
):

    mock_put = mocker.patch(
        "garden_ai.backend_client.BackendClient._put",
        return_value=garden_nested_metadata_json,
    )

    garden = garden_client._create_garden(mock_GardenMetadata)

    mock_put.assert_called_once()
    assert garden.metadata.doi == mock_GardenMetadata.doi


def test_create_garden_raises_on_failure(
    garden_client,
    mocker,
    mock_GardenMetadata,
):
    # Simulate a failure of the backend
    mock_put = mocker.patch(
        "garden_ai.backend_client.BackendClient._put",
        side_effect=Exception("Intentional Error for Testing."),
    )

    with pytest.raises(Exception):
        garden = garden_client.create_garden(mock_GardenMetadata)


def test_add_entrypoint_to_garden_raises_on_duplicate_entrypoint_names(
    garden_client,
    mocker,
    garden_nested_metadata_json,
    entrypoint_metadata_json,
):

    garden_doi = garden_nested_metadata_json["doi"]
    entrypoint_doi = entrypoint_metadata_json["doi"]

    # Add an entrypoint alias
    alias = "my_alias"
    garden_nested_metadata_json["entrypoint_aliases"][entrypoint_doi] = alias
    entrypoint_metadata_json["short_name"] = alias

    # Mock GET responses from the backend
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        side_effect=[garden_nested_metadata_json, entrypoint_metadata_json],
    )

    # Mock the PUT to the backend
    mocker.patch(
        "garden_ai.backend_client.BackendClient._put",
        return_value=garden_nested_metadata_json,
    )

    with pytest.raises(ValueError) as e:
        # Addidg an alias that already exists should raise
        garden_client.add_entrypoint_to_garden(entrypoint_doi, garden_doi, alias)

    assert "garden already has another entrypoint under that name" in e.value.args[0]


def test_add_entrypoint_to_garden_raises_on_invalid_identifier(
    garden_client,
    mocker,
    garden_nested_metadata_json,
    entrypoint_metadata_json,
):
    garden_doi = garden_nested_metadata_json["doi"]
    entrypoint_doi = entrypoint_metadata_json["doi"]

    # Mock GET responses from the backend
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        side_effect=[garden_nested_metadata_json, entrypoint_metadata_json],
    )

    # Mock the PUT to the backend
    mocker.patch(
        "garden_ai.backend_client.BackendClient._put",
        return_value=garden_nested_metadata_json,
    )

    # Set an invalid identifier
    bad_alias = "not a valid identifier"

    with pytest.raises(AssertionError) as e:
        # Addidg an alias that already exists should raise
        garden_client.add_entrypoint_to_garden(entrypoint_doi, garden_doi, bad_alias)

    assert "must be a valid python identifier" in e.value.args[0]


def test_add_entrypoint_to_garden_posts_updated_garden_to_backend(
    garden_client,
    mocker,
    garden_nested_metadata_json,
    entrypoint_metadata_json,
):
    new_doi = "some/doi"
    garden_doi = garden_nested_metadata_json["doi"]
    entrypoint_doi = entrypoint_metadata_json["doi"] = new_doi

    # Mock GET responses from the backend
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        side_effect=[garden_nested_metadata_json, entrypoint_metadata_json],
    )

    # Mock the PUT to the backend
    mock_put = mocker.patch(
        "garden_ai.backend_client.BackendClient._put",
        return_value=garden_nested_metadata_json,
    )

    _ = garden_client.add_entrypoint_to_garden(entrypoint_doi, garden_doi)

    mock_put.assert_called_once()


def test_register_garden_doi_updates_datacite(
    garden_client,
    mocker,
    garden_nested_metadata_json,
):
    # Patch the backend methods that make network calls
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value=garden_nested_metadata_json,
    )
    mocker.patch(
        "garden_ai.backend_client.BackendClient._put",
        return_value=garden_nested_metadata_json,
    )

    # Mock the actual datacite update
    mock_update_datacite = mocker.patch(
        "garden_ai.client.GardenClient._update_datacite"
    )

    doi = garden_nested_metadata_json["doi"]
    garden_client.register_garden_doi(doi)

    mock_update_datacite.assert_called_once()


def test_register_garden_doi_updates_backend(
    garden_client,
    mocker,
    garden_nested_metadata_json,
):
    # Patch the backend methods that make network calls
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value=garden_nested_metadata_json,
    )
    # Mock the datacite update
    mocker.patch("garden_ai.client.GardenClient._update_datacite")

    # This is the call we are looking for
    mock_put = mocker.patch(
        "garden_ai.backend_client.BackendClient._put",
        return_value=garden_nested_metadata_json,
    )

    doi = garden_nested_metadata_json["doi"]
    garden_client.register_garden_doi(doi)

    mock_put.assert_called_once()


def test_register_garden_doi_raises_on_backend_failure(
    garden_client,
    mocker,
    garden_nested_metadata_json,
):
    # Patch the backend methods that make network calls
    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value=garden_nested_metadata_json,
    )

    # Simulate a failure from the backend
    mock_datacite_update = mocker.patch(
        "garden_ai.backend_client.BackendClient.update_doi_on_datacite",
        side_effect=Exception("Intention Error for Testing."),
    )

    mock_put = mocker.patch(
        "garden_ai.backend_client.BackendClient._put",
    )

    doi = garden_nested_metadata_json["doi"]

    with pytest.raises(Exception):
        garden_client.register_garden_doi(doi)

    mock_put.assert_not_called()


def test_register_entrypoint_doi_updates_datacite(
    garden_client,
    mocker,
    entrypoint_metadata_json,
):

    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value=entrypoint_metadata_json,
    )

    mock_datacite_update = mocker.patch(
        "garden_ai.client.GardenClient._update_datacite",
    )

    mocker.patch(
        "garden_ai.backend_client.BackendClient._put",
        return_value=entrypoint_metadata_json,
    )

    doi = entrypoint_metadata_json["doi"]
    garden_client.register_entrypoint_doi(doi)

    mock_datacite_update.assert_called_once()


def test_register_entrypoint_doi_raises_on_backend_failure(
    garden_client,
    mocker,
    entrypoint_metadata_json,
):

    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value=entrypoint_metadata_json,
    )

    mock_put = mocker.patch(
        "garden_ai.backend_client.BackendClient._put",
    )

    mock_update_on_datacite = mocker.patch(
        "garden_ai.backend_client.BackendClient.update_doi_on_datacite",
        side_effect=Exception("Intentional Error for Testing"),
    )

    doi = entrypoint_metadata_json["doi"]

    with pytest.raises(Exception):
        garden_client.register_entrypoint_doi(doi)

    mock_update_on_datacite.assert_called_once()
    mock_put.assert_not_called()


def test_register_entrypoint_doi_updates_backend(
    garden_client,
    mocker,
    entrypoint_metadata_json,
):

    mocker.patch(
        "garden_ai.backend_client.BackendClient._get",
        return_value=entrypoint_metadata_json,
    )

    mocker.patch(
        "garden_ai.client.GardenClient._update_datacite",
    )

    mock_put = mocker.patch(
        "garden_ai.backend_client.BackendClient._put",
        return_value=entrypoint_metadata_json,
    )

    doi = entrypoint_metadata_json["doi"]
    garden_client.register_entrypoint_doi(doi)

    mock_put.assert_called_once()


def test_init_fails_if_auth_fails(
    mocker,
    patch_garden_constants,
):
    # Simulate an authentication error
    mocker.patch(
        "garden_ai.client.GardenClient._create_authorizer",
        side_effect=Exception("Intentional Error for Testing"),
    )

    with pytest.raises(Exception):
        client = GardenClient()
