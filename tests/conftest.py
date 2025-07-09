from copy import deepcopy
import json
import pathlib  # noqa
from unittest.mock import patch

from globus_compute_sdk.sdk.login_manager.manager import LoginManager  # type: ignore
from globus_sdk import AuthLoginClient, OAuthTokenResponse
import pytest
from typer.testing import CliRunner

from garden_ai.backend_client import BackendClient
from garden_ai.client import GardenClient
from garden_ai.constants import GardenConstants
from garden_ai.garden_file_adapter import GardenFileAdapter
from garden_ai.schemas.garden import GardenMetadata


def pytest_addoption(parser, pluginmanager):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )


def pytest_collection_modifyitems(session, config, items):
    skip_integration_tests = True
    skip_ete_tests = True

    # run integration tests when pytest is called with --integration
    if config.getoption("--integration"):
        skip_integration_tests = False

    if marker := config.getoption("-m"):
        if "integration" in marker and "not integration" not in marker:
            # if -m "integration" is given on the cli, don't skip integration tests
            skip_integration_tests = False

        if "ete" in marker and "not e2e" not in marker:
            skip_ete_tests = False

    skip_integration = pytest.mark.skip(
        "need -m 'integration' or --integration option to run"
    )
    skip_ete = pytest.mark.skip("need -m 'ete' to run")

    for item in items:
        if skip_integration_tests and "integration" in item.keywords:
            item.add_marker(skip_integration)

        if skip_ete_tests and "ete" in item.keywords:
            item.add_marker(skip_ete)


@pytest.fixture(scope="session")
def cli_runner() -> CliRunner:
    """Return a typer.testing.CliRunner for use in tests."""
    return CliRunner()


@pytest.fixture
def app(garden_client, mocker):
    """Provides an instance of the garden-ai CLI app for tests.

    Overrides the GardenClient that the app constructs in commands/subcommands.
    Replaces the GardenClient with one that has mocked auth and network functions.
    """
    with patch("garden_ai.GardenClient", return_value=garden_client):
        from garden_ai.app.main import app as cli_app

        yield cli_app


@pytest.fixture
def mock_authorizer_tuple(mocker):
    """Return mocks of the Globus authorizer for the GardenClient."""
    mock_authorizer = mocker.Mock()
    mock_authorizer_constructor = mocker.patch(
        "garden_ai.client.RefreshTokenAuthorizer", return_value=mock_authorizer
    )
    return mock_authorizer_constructor, mock_authorizer


@pytest.fixture
def identity_jwt() -> str:
    """Return a mock Globus Auth JWT"""
    return "eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2YzllMjIzZi1jMjE1LTRjMjYtOWFiYi0yNjJkYmNlMDAwMWMiLCJlbWFpbCI6IndpbGxlbmdsZXJAdWNoaWNhZ28uZWR1IiwibGFzdF9hdXRoZW50aWNhdGlvbiI6MTY3NjU2MjEyNiwiaWRlbnRpdHlfc2V0IjpbeyJzdWIiOiI2YzllMjIzZi1jMjE1LTRjMjYtOWFiYi0yNjJkYmNlMDAwMWMiLCJlbWFpbCI6IndpbGxlbmdsZXJAdWNoaWNhZ28uZWR1IiwibGFzdF9hdXRoZW50aWNhdGlvbiI6MTY3NjU2MjEyNn1dLCJpc3MiOiJodHRwczovL2F1dGguZ2xvYnVzLm9yZyIsImF1ZCI6ImNmOWY4OTM4LWZiNzItNDM5Yy1hNzBiLTg1YWRkZjFiODUzOSIsImV4cCI6MTY3Nzc4NzMzNCwiaWF0IjoxNjc3NjE0NTM0LCJhdF9oYXNoIjoiT1VMX0s3ZmVyNXNBdk03cEI0dXJnNE95dXZVOUxEcGh3SnhCX1VUXzY1cyJ9.TssuMsFeKFQH9Bd29c2Bj0V_f-KkN8alGtHnZOZbg5AQwVPokthHRA4bro7IHuOCXIoh3kX03KPNcLfyRRM5JN1d4SKl0L9KpkJB45BkKzKcg8KPgChOzs_9jRiiDwmXvIpgWiUNVI4grHIEYVpB_VdFdKw6EwWQgu6ZrN_2rvpa45Pc-NZ_-WR4WDFAx2Hak7sXRXslY_1ftlRgV9348uwp78jh1pnXft-mpgpzwqHVALLKgzecCESYmaipWTd3-atczpH9SPIxOn7DoiX2I2Nhn_8IkrhIZnbmtOyY7wINrSGFonN49AncVTNq9AfIngZB26spUByHW4mLB6E2Mw"  # noqa: E501


@pytest.fixture
def token() -> dict:
    """Return a dict with refresh_token, access_token, and expires_at_seconds fields."""
    return {
        "refresh_token": "MyRefreshToken",
        "access_token": "MyAccessToken",
        "expires_at_seconds": 1024,
    }


@pytest.fixture
def mock_keystore(mocker):
    """Mock the GardenFileAdapter in to have a ketstore for tests"""
    mock_keystore = mocker.MagicMock(GardenFileAdapter)
    mocker.patch("garden_ai.client.GardenFileAdapter").return_value = mock_keystore
    return mock_keystore


@pytest.fixture
def patch_backend_client_requests(mocker, garden_nested_metadata_json) -> None:
    """Patches BackendClient methods that make network requests."""
    backend_client = "garden_ai.backend_client.BackendClient"
    mocker.patch(
        f"{backend_client}.get_user_info",
        return_value={"email": "fake@email.com", "identity_id": "ugvhbkjn"},
    )


@pytest.fixture
def patch_garden_constants(mocker, tmp_path):
    """Patches fields in GardenConstants with temp values for tests."""
    mocker.patch.object(GardenConstants, "GARDEN_DIR", tmp_path)
    mocker.patch.object(GardenConstants, "GARDEN_KEY_STORE", tmp_path / "tokens.json")
    yield


@pytest.fixture
def garden_client(
    mocker,
    mock_authorizer_tuple,
    mock_keystore,
    token,
    identity_jwt,
    patch_garden_constants,
    mock_user_info_response,
) -> GardenClient:
    """Return a GardenClient with mocked auth credentials."""

    # mocker, mock_authorizer_tuple, token, mock_keystore
    mock_authorizer_constructor, mock_authorizer = mock_authorizer_tuple

    # Mocks for Login Flow
    mock_auth_client = mocker.MagicMock(AuthLoginClient)
    mock_auth_client.oauth2_get_authorize_url = mocker.Mock(
        return_value="https://globus.auth.garden"
    )
    mock_auth_client.oauth2_start_flow = mocker.Mock()
    # mocker.patch("garden_ai.client.input").return_value = "my token"
    mocker.patch("garden_ai.client.Prompt.ask").return_value = "my token"
    mocker.patch("garden_ai.client.typer.launch")

    mock_token_response = mocker.MagicMock(OAuthTokenResponse)
    mock_token_response.data = {"id_token": identity_jwt}
    mock_token_response.by_resource_server = {
        "groups.api.globus.org": token,
        "0948a6b0-a622-4078-b0a4-bfd6d77d65cf": token,
        "funcx_service": token,
        "auth.globus.org": token,
    }
    mock_auth_client.oauth2_exchange_code_for_tokens = mocker.Mock(
        return_value=mock_token_response
    )

    # Mocks compute client login
    mock_login_manager = mocker.MagicMock(LoginManager)
    mock_login_manager.ensure_logged_in = mocker.Mock(return_value=True)
    mocker.patch("globus_compute_sdk.sdk.client.LoginManager").return_value = (
        mock_login_manager
    )

    mocker.patch(
        "garden_ai.client.GardenClient._do_login_flow", return_value=mock_token_response
    )

    mocker.patch(
        "garden_ai.backend_client.BackendClient.get_user_info",
        return_value=mock_user_info_response,
    )
    # Call the Garden constructor
    gc = GardenClient(auth_client=mock_auth_client)
    return gc


@pytest.fixture
def logged_in_user(tmp_path):
    """Simulates a logged-in user by creating a temporary GARDEN_KEY_STORE file."""
    with patch.object(
        GardenConstants,
        "GARDEN_KEY_STORE",
        tmp_path,
    ):
        yield


@pytest.fixture(scope="session")
def garden_nested_metadata_raw() -> dict:
    """Return a dict with a valid GardenMetadata schema with nested functions."""
    f = (
        pathlib.Path(__file__).parent
        / "fixtures"
        / "garden_nested_metadata_response.json"
    )
    with open(f, "r") as f_in:
        return json.load(f_in)


@pytest.fixture
def garden_nested_metadata_json(garden_nested_metadata_raw) -> dict:
    """Return a dict with a valid GardenMetadata schema with nested functions."""
    return deepcopy(garden_nested_metadata_raw)


@pytest.fixture(scope="session")
def garden_nested_metadata_raw_with_modal_class() -> dict:
    """Return a dict with a valid GardenMetadata schema with nested functions."""
    f = (
        pathlib.Path(__file__).parent
        / "fixtures"
        / "garden_nested_metadata_with_modal_class.json"
    )
    with open(f, "r") as f_in:
        return json.load(f_in)


@pytest.fixture
def garden_nested_metadata_json_with_modal_class(
    garden_nested_metadata_raw_with_modal_class,
) -> dict:
    """Return a dict with a valid GardenMetadata schema with nested functions."""
    return deepcopy(garden_nested_metadata_raw_with_modal_class)


@pytest.fixture(scope="session")
def modal_function_metadata_raw() -> dict:
    """Return a dict with a valid ModalFunctionMetadata schema."""
    f = pathlib.Path(__file__).parent / "fixtures" / "modal_function_metadata.json"
    with open(f, "r") as f_in:
        return json.load(f_in)


@pytest.fixture
def modal_function_metadata_json(modal_function_metadata_raw) -> dict:
    """Return a dict with a valid ModalFunctionMetadata schema."""
    return deepcopy(modal_function_metadata_raw)


@pytest.fixture(scope="session")
def modal_method_metadata_raw() -> dict:
    """Return a dict with a valid ModalFunctionMetadata schema,
    where the function in question is a class method."""
    f = pathlib.Path(__file__).parent / "fixtures" / "modal_class_metadata.json"
    with open(f, "r") as f_in:
        return json.load(f_in)


@pytest.fixture
def modal_method_metadata_json(modal_method_metadata_raw) -> dict:
    """Return a dict with a valid ModalFunctionMetadata schema."""
    return deepcopy(modal_method_metadata_raw)


@pytest.fixture
def mock_GardenMetadata(garden_nested_metadata_json) -> GardenMetadata:
    """Return a GardenMetadata object populated with test data."""
    return GardenMetadata(
        **garden_nested_metadata_json,
    )


@pytest.fixture
def mock_user_info_response(faker) -> dict:
    """Return dict of fake user info like we get from the backend /users route"""
    return {
        "username": faker.user_name(),
        "name": faker.name(),
        "email": faker.email(),
        "phone_number": faker.phone_number(),
        "affiliations": [
            faker.name(),
        ],
        "skills": [
            faker.first_name(),
            faker.first_name(),
        ],
        "domains": [
            faker.first_name(),
            faker.first_name(),
        ],
        "profile_pic_id": 1,
        "identity_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "saved_garden_dois": ["10.23677/fake-doi"],
    }


@pytest.fixture
def backend_client(garden_client):
    """Return a BackendClient instance"""
    return BackendClient(garden_client.garden_authorizer)
