from typing import List

import pytest
import os
from globus_compute_sdk import Client  # type: ignore
from globus_sdk import AuthClient, OAuthTokenResponse, SearchClient
from globus_sdk.tokenstorage import SimpleJSONFileAdapter
from mlflow.pyfunc import PyFuncModel  # type: ignore

import garden_ai
from garden_ai import Garden, GardenClient, Pipeline, step
from garden_ai.pipelines import RegisteredPipeline
from garden_ai.mlmodel import RegisteredModel, DatasetConnection
from tests.fixtures.helpers import get_fixture_file_path  # type: ignore


@pytest.fixture(autouse=True)
def do_not_sleep(mocker):
    mocker.patch("time.sleep")


@pytest.fixture(autouse=True)
def auto_mock_GardenClient_set_up_mlflow_env(mocker):
    mocker.patch("garden_ai.client.GardenClient._set_up_mlflow_env")


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
    mock_keystore = mocker.MagicMock(SimpleJSONFileAdapter)
    mocker.patch("garden_ai.client.SimpleJSONFileAdapter").return_value = mock_keystore
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
def pipeline_toy_example(tmp_requirements_txt):
    # define a step using the decorator
    @step(authors=["Sister Constance"])
    def split_peas(ps: List) -> List[tuple]:
        return [(p / 2, p / 2) for p in ps]

    class Soup:
        ...

    @step(authors=["Friar Hugo"])
    def make_soup(splits: List[tuple]) -> Soup:
        return Soup()

    @step(authors=["Abbot Mortimer"], input_info="a spoonful of Soup object")
    def rate_soup(soup_sample: Soup) -> float:
        return 10 / 10

    rate_soup.contributors += ["Friar Hugo", "Sister Constance"]

    pea_edibility_pipeline = Pipeline(
        title="Pea Edibility Pipeline",
        steps=[split_peas, make_soup, rate_soup],
        authors=["Brian Jacques"],
        description="A pipeline for perfectly-reproducible soup ratings.",
        requirements_file=str(tmp_requirements_txt),
        doi="10.26311/fake-doi",
    )

    # the complete pipeline is now also callable by itself
    assert pea_edibility_pipeline([1, 2, 3]) == 10 / 10
    return pea_edibility_pipeline


@pytest.fixture
def registered_pipeline_toy_example(pipeline_toy_example, noop_func_uuid):
    pipeline_toy_example.doi = "10.26311/fake-doi"
    pipeline_toy_example.func_uuid = noop_func_uuid
    pipeline_toy_example.short_name = "pipeline_toy_example"
    return RegisteredPipeline.from_pipeline(pipeline_toy_example)


@pytest.fixture
def garden_all_fields(registered_pipeline_toy_example):
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
    pea_garden.validate()
    return pea_garden


@pytest.fixture
def tmp_requirements_txt(tmp_path):
    """
    Fixture that creates a temporary requirements.txt file
    """
    contents = "Flask==2.1.1\npandas>=1.3.0\nnumpy==1.21.2\nscikit-learn>=0.24.2\n"
    file_path = tmp_path / "requirements.txt"
    with open(file_path, "w") as f:
        f.write(contents)
    return file_path


@pytest.fixture
def tmp_conda_yml(tmp_path):
    """
    Fixture that creates a temporary `conda.yml` file.
    """
    contents = """\
name: my_env
channels:
- defaults
dependencies:
- python=3.8
- flask=2.1.1
- pandas>=1.3.0
- pip:
    - mlflow<3,>=2.2
    - cloudpickle==2.2.1
    - numpy==1.23.5
    - psutil==5.9.4
    - scikit-learn==1.2.2
    - fake-package==9.9.9
"""
    file_path = tmp_path / "conda.yml"
    with open(file_path, "w") as f:
        f.write(contents)
    return file_path


@pytest.fixture
def step_with_model(mocker, tmp_conda_yml):
    mock_model = mocker.MagicMock(PyFuncModel)
    mocker.patch("garden_ai.mlmodel.load_model").return_value = mock_model
    mocker.patch(
        "garden_ai.mlmodel.mlflow.pyfunc.get_model_dependencies"
    ).return_value = tmp_conda_yml

    @step
    def uses_model_in_default(
        arg: object,
        default_arg_model: object = garden_ai.Model(
            "email@addr.ess-fake-model/fake-version"
        ),
    ) -> object:
        pass

    return uses_model_in_default


@pytest.fixture
def pipeline_using_step_with_model(mocker, tmp_requirements_txt, step_with_model):
    # define a step using the decorator
    @step(authors=["Sister Constance"])
    def split_peas(ps: List) -> List[tuple]:
        return [(p / 2, p / 2) for p in ps]

    class Soup:
        ...

    @step(authors=["Friar Hugo"])
    def make_soup(splits: List[tuple]) -> Soup:
        return Soup()

    ALL_STEPS = (split_peas, make_soup, step_with_model)  # see fixture

    pea_edibility_pipeline = Pipeline(
        title="Pea Edibility Pipeline",
        steps=ALL_STEPS,
        authors=["Brian Jacques"],
        description="A pipeline for perfectly-reproducible soup ratings.",
        requirements_file=str(tmp_requirements_txt),
        doi="10.26311/fake-doi",
    )

    return pea_edibility_pipeline


@pytest.fixture
def database_with_unconnected_pipeline(tmp_path):
    source_path = get_fixture_file_path(
        "database_dumps/one_pipeline_one_garden_unconnected.txt"
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
        "database_dumps/one_pipeline_one_garden_connected.txt"
    )
    with open(source_path, "r") as file:
        contents = file.read()
    data_file = tmp_path / "data.json"
    with open(data_file, "w") as f:
        f.write(contents)
    return tmp_path


@pytest.fixture
def database_with_model(tmp_path):
    source_path = get_fixture_file_path("database_dumps/one_model.json")
    with open(source_path, "r") as file:
        contents = file.read()
    data_file = tmp_path / "data.json"
    with open(data_file, "w") as f:
        f.write(contents)
    return tmp_path


@pytest.fixture
def second_draft_of_model():
    return RegisteredModel(
        model_name="unit-test-model",
        version="2",
        user_email="test@example.com",
        flavor="sklearn",
        connections=[
            DatasetConnection(
                **{
                    "type": "dataset",
                    "relationship": "origin",
                    "doi": "10.18126/wg3u-g8vu",
                    "repository": "Foundry",
                    "url": "https://foundry-ml.org/#/datasets/10.18126%2Fwg3u-g8vu",
                }
            )
        ],
    )


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
