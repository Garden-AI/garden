from typing import List

import pytest
from garden_ai import Garden, GardenClient, Pipeline, step
from globus_sdk import AuthClient, OAuthTokenResponse
from globus_sdk.tokenstorage import SimpleJSONFileAdapter


@pytest.fixture
def mock_authorizer_tuple(mocker):
    mock_authorizer = mocker.Mock()
    mock_authorizer_constructor = mocker.patch(
        "garden_ai.garden.RefreshTokenAuthorizer", return_value=mock_authorizer
    )
    return mock_authorizer_constructor, mock_authorizer


@pytest.fixture
def token():
    return {
        "refresh_token": "MyRefreshToken",
        "access_token": "MyAccessToken",
        "expires_at_seconds": 1024,
    }


@pytest.fixture
def mock_keystore(mocker):
    mock_keystore = mocker.MagicMock(SimpleJSONFileAdapter)
    mocker.patch("garden_ai.garden.SimpleJSONFileAdapter").return_value = mock_keystore
    return mock_keystore


@pytest.fixture
def garden_client(mocker, mock_authorizer_tuple, mock_keystore, token):
    # blindly stolen from test_garden.py test

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
    mocker.patch("garden_ai.garden.input").return_value = "my token"

    mock_token_response = mocker.MagicMock(OAuthTokenResponse)
    mock_token_response.by_resource_server = {"groups.api.globus.org": token}
    mock_auth_client.oauth2_exchange_code_for_tokens = mocker.Mock(
        return_value=mock_token_response
    )

    # Call the Garden constructor
    gc = GardenClient(auth_client=mock_auth_client)
    return gc


@pytest.fixture
def garden_all_fields(garden_client):
    client = garden_client
    pea_garden: Garden = client.create_garden(
        authors=["Mendel, Gregor"],
        title="Experiments on Plant Hybridization",
        contributors=["St. Thomas Abbey"],
    )
    pea_garden.year = "1863"
    pea_garden.language = "en"
    pea_garden.version = "0.0.1"
    pea_garden.description = "This Garden houses ML pipelines for Big Pea Data."
    pea_garden.doi = pea_garden._doi_prefix + "/fake-doi"
    assert pea_garden.pipelines is not None
    return pea_garden


@pytest.fixture
def garden_title_authors_doi_only(garden_client):
    pea_garden = garden_client.create_garden(
        authors=["Mendel, Gregor"], title="Experiments on Plant Hybridization"
    )
    pea_garden.doi = pea_garden._doi_prefix + "/fake-doi"
    return pea_garden


@pytest.fixture
def garden_no_fields(garden_client):
    return garden_client.create_garden()


@pytest.fixture
def pipeline_toy_example():
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
        title="Concept: a pipeline for soup",
        steps=[split_peas, make_soup, rate_soup],
        authors=["Brian Jacques"],
    )

    # the complete pipeline is now also callable by itself
    assert pea_edibility_pipeline([1, 2, 3]) == 10 / 10
    return pea_edibility_pipeline


def pipeline_sklearn_example():
    # TODO
    pass
