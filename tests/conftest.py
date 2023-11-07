import os

import pytest
from globus_compute_sdk import Client  # type: ignore
from globus_compute_sdk.sdk.login_manager.manager import LoginManager  # type: ignore
from globus_sdk import AuthClient, OAuthTokenResponse, SearchClient

from garden_ai import Garden, GardenClient
from garden_ai.garden_file_adapter import GardenFileAdapter
from garden_ai.pipelines import RegisteredPipeline
from tests.fixtures.helpers import get_fixture_file_path  # type: ignore


@pytest.fixture(autouse=True)
def do_not_sleep(mocker):
    mocker.patch("time.sleep")


@pytest.fixture(autouse=True)
def auto_mock_GardenClient_mint_draft_doi(mocker):
    mocker.patch(
        "garden_ai.client.GardenClient._mint_draft_doi",
        return_value="10.26311/fake-doi",
    )


@pytest.fixture
def mock_authorizer_tuple(mocker):
    mock_authorizer = mocker.Mock()
    mock_authorizer_constructor = mocker.patch(
        "garden_ai.client.RefreshTokenAuthorizer", return_value=mock_authorizer
    )
    return mock_authorizer_constructor, mock_authorizer


@pytest.fixture
def identity_jwt():
    return "eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2YzllMjIzZi1jMjE1LTRjMjYtOWFiYi0yNjJkYmNlMDAwMWMiLCJlbWFpbCI6IndpbGxlbmdsZXJAdWNoaWNhZ28uZWR1IiwibGFzdF9hdXRoZW50aWNhdGlvbiI6MTY3NjU2MjEyNiwiaWRlbnRpdHlfc2V0IjpbeyJzdWIiOiI2YzllMjIzZi1jMjE1LTRjMjYtOWFiYi0yNjJkYmNlMDAwMWMiLCJlbWFpbCI6IndpbGxlbmdsZXJAdWNoaWNhZ28uZWR1IiwibGFzdF9hdXRoZW50aWNhdGlvbiI6MTY3NjU2MjEyNn1dLCJpc3MiOiJodHRwczovL2F1dGguZ2xvYnVzLm9yZyIsImF1ZCI6ImNmOWY4OTM4LWZiNzItNDM5Yy1hNzBiLTg1YWRkZjFiODUzOSIsImV4cCI6MTY3Nzc4NzMzNCwiaWF0IjoxNjc3NjE0NTM0LCJhdF9oYXNoIjoiT1VMX0s3ZmVyNXNBdk03cEI0dXJnNE95dXZVOUxEcGh3SnhCX1VUXzY1cyJ9.TssuMsFeKFQH9Bd29c2Bj0V_f-KkN8alGtHnZOZbg5AQwVPokthHRA4bro7IHuOCXIoh3kX03KPNcLfyRRM5JN1d4SKl0L9KpkJB45BkKzKcg8KPgChOzs_9jRiiDwmXvIpgWiUNVI4grHIEYVpB_VdFdKw6EwWQgu6ZrN_2rvpa45Pc-NZ_-WR4WDFAx2Hak7sXRXslY_1ftlRgV9348uwp78jh1pnXft-mpgpzwqHVALLKgzecCESYmaipWTd3-atczpH9SPIxOn7DoiX2I2Nhn_8IkrhIZnbmtOyY7wINrSGFonN49AncVTNq9AfIngZB26spUByHW4mLB6E2Mw"  # noqa: E501


@pytest.fixture
def token():
    return {
        "refresh_token": "MyRefreshToken",
        "access_token": "MyAccessToken",
        "expires_at_seconds": 1024,
    }


@pytest.fixture
def noop_func_uuid():
    return "9f5688ac-424d-443e-b525-97c72e4e083f"


@pytest.fixture
def mock_keystore(mocker):
    mock_keystore = mocker.MagicMock(GardenFileAdapter)
    mocker.patch("garden_ai.client.GardenFileAdapter").return_value = mock_keystore
    return mock_keystore


@pytest.fixture
def garden_client(mocker, mock_authorizer_tuple, mock_keystore, token, identity_jwt):
    # blindly stolen from test_client.py test

    # mocker, mock_authorizer_tuple, token, mock_keystore
    mock_authorizer_constructor, mock_authorizer = mock_authorizer_tuple
    # Mocks for KeyStore
    mock_keystore.file_exists.return_value = False

    # Mocks for Login Flow
    mock_auth_client = mocker.MagicMock(AuthClient)
    mock_auth_client.oauth2_get_authorize_url = mocker.Mock(
        return_value="https://globus.auth.garden"
    )
    mock_auth_client.oauth2_start_flow = mocker.Mock()
    # mocker.patch("garden_ai.client.input").return_value = "my token"
    mocker.patch("garden_ai.client.Prompt.ask").return_value = "my token"
    mocker.patch("garden_ai.client.typer.launch")
    mock_search_client = mocker.MagicMock(SearchClient)

    mock_token_response = mocker.MagicMock(OAuthTokenResponse)
    mock_token_response.data = {"id_token": identity_jwt}
    mock_token_response.by_resource_server = {
        "groups.api.globus.org": token,
        "search.api.globus.org": token,
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
    mocker.patch(
        "globus_compute_sdk.sdk.client.LoginManager"
    ).return_value = mock_login_manager

    # Call the Garden constructor
    gc = GardenClient(auth_client=mock_auth_client, search_client=mock_search_client)
    return gc


@pytest.fixture
def compute_client(mocker):
    mock_compute_client = mocker.MagicMock(Client)
    mock_compute_client.build_container = mocker.Mock(
        return_value="d1fc6d30-be1c-4ac4-a289-d87b27e84357"
    )
    mock_compute_client.get_container_build_status = mocker.Mock(return_value="ready")
    mock_compute_client.register_function = mocker.Mock(
        return_value="f9072604-6e71-4a14-a336-f7fc4a677293"
    )
    return mock_compute_client


@pytest.fixture
def garden_title_authors_doi_only():
    pea_garden = Garden(
        authors=["Mendel, Gregor"],
        title="Experiments on Plant Hybridization",
        doi="10.26311/fake-doi",
    )
    return pea_garden


@pytest.fixture
def garden_no_fields():
    return Garden(doi="10.26311/fake-doi")  # DOI is required to instantiate


@pytest.fixture
def registered_pipeline_toy_example(noop_func_uuid):
    return RegisteredPipeline(
        title="Pea Edibility Pipeline",
        short_name="pipeline_toy_example",
        authors=["Brian Jacques"],
        description="A pipeline for perfectly-reproducible soup ratings.",
        container_uuid="4f5688ac-424d-443e-b525-97c72e4e013f",
        func_uuid=noop_func_uuid,
        doi="10.26311/fake-doi",
    )


@pytest.fixture
def garden_all_fields(mocker, registered_pipeline_toy_example):
    mocker.patch(
        "garden_ai.Garden._collect_pipelines",
        return_value=[registered_pipeline_toy_example],
    )

    pea_garden = Garden(
        authors=["Mendel, Gregor"],
        title="Experiments on Plant Hybridization",
        contributors=["St. Thomas Abbey"],
        doi="10.26311/fake-doi",
    )
    pea_garden.year = "1863"
    pea_garden.language = "en"
    pea_garden.version = "0.0.1"
    pea_garden.description = "This Garden houses ML pipelines for Big Pea Data."
    pea_garden.pipeline_ids += [registered_pipeline_toy_example.doi]
    return pea_garden


@pytest.fixture
def database_with_unconnected_pipeline(tmp_path):
    source_path = get_fixture_file_path(
        "database_dumps/one_pipeline_one_garden_unconnected.json"
    )
    with open(source_path, "r") as file:
        contents = file.read()
    data_file = tmp_path / "data.json"
    with open(data_file, "w") as f:
        f.write(contents)
    return tmp_path


@pytest.fixture
def database_with_connected_pipeline(tmp_path):
    source_path = get_fixture_file_path(
        "database_dumps/one_pipeline_one_garden_connected.json"
    )
    with open(source_path, "r") as file:
        contents = file.read()
    data_file = tmp_path / "data.json"
    with open(data_file, "w") as f:
        f.write(contents)
    return tmp_path


@pytest.fixture
def cc_grant_tuple():
    client_id = os.getenv("GARDEN_API_CLIENT_ID")
    client_secret = os.getenv("GARDEN_API_CLIENT_SECRET")
    return (client_id, client_secret)


# Fixture JSON responses from the Search API
def read_fixture_text(fname):
    path = get_fixture_file_path(fname)
    with open(path, "r") as f:
        return f.read()


@pytest.fixture
def valid_search_by_subject():
    return read_fixture_text("search_results/valid_search_by_subject.json")


@pytest.fixture
def valid_search_by_doi():
    return read_fixture_text("search_results/valid_search_by_doi.json")


@pytest.fixture
def empty_search_by_doi():
    return read_fixture_text("search_results/empty_search_by_doi.json")


@pytest.fixture
def path_to_pipeline_with_main_block():
    return get_fixture_file_path("fixture_pipeline/pipeline.py")
