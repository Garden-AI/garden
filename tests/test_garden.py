import pytest

from garden_ai import GardenClient, Garden, Pipeline, step
from globus_sdk import AuthClient, OAuthTokenResponse, AuthAPIError
from globus_sdk.tokenstorage import SimpleJSONFileAdapter

from garden_ai.garden import AuthException
from pydantic import ValidationError
from typing import Union


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
    # blindly stolen from tests below

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
    pea_garden = garden_client.create_garden(
        authors=["Mendel, Gregor"], title="Experiments on Plant Hybridization"
    )
    pea_garden.authors += ["St Thomas' Abbey"]
    pea_garden.year = 1863
    pea_garden.description = """This Garden houses sophisticated ML pipelines
                                for Big Pea Data extraction and classification.
                                It consists of a 2-hectare plot behind the monastery,
                                and a 30,000-plant dataset."""
    pea_garden.language = "en"
    pea_garden.doi = "10.55555/fake-doi"
    pea_garden.version = "0.0.1"
    return pea_garden


@pytest.fixture
def garden_required_fields_only(garden_client):
    pea_garden = garden_client.create_garden(
        authors=["Mendel, Gregor"], title="Experiments on Plant Hybridization"
    )
    # pea_garden.doi = "10.55555/fake-doi"
    return pea_garden


@pytest.fixture
def garden_no_fields(garden_client):
    return garden_client.create_garden()


def test_client_no_previous_tokens(mocker, mock_authorizer_tuple, token, mock_keystore):
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

    assert gc.auth_key_store == mock_keystore
    mock_auth_client.oauth2_exchange_code_for_tokens.assert_called_with("my token")

    mock_auth_client.oauth2_start_flow.assert_called_with(
        refresh_tokens=True,
        requested_scopes="urn:globus:auth:scope:groups.api.globus.org:view_my_groups_and_memberships",
    )

    mock_keystore.store.assert_called_with(mock_token_response)
    mock_authorizer_constructor.assert_called_with(
        "MyRefreshToken",
        mock_auth_client,
        access_token="MyAccessToken",
        expires_at=1024,
        on_refresh=mock_keystore.on_refresh,
    )

    assert gc.authorizer == mock_authorizer


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

    assert gc.authorizer == mock_authorizer


def test_client_invalid_auth_token(mocker, mock_authorizer_tuple, token, mock_keystore):
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


def test_create_empty_garden(garden_client):
    # feels silly, but we do want users to be able to initialize an empty garden
    # & fill in required stuff later

    # object should exist with default-illegal fields
    garden = garden_client.create_garden()

    assert not garden.authors
    assert not garden.title
    assert not garden.doi


def test_validate_all_fields(garden_all_fields):
    garden_all_fields.validate()


def test_validate_no_fields(garden_no_fields):
    with pytest.raises(ValidationError):
        garden_no_fields.validate()


def test_validate_required_only(garden_required_fields_only):
    garden = garden_required_fields_only
    assert not garden.doi
    with pytest.raises(ValidationError):
        garden.validate()
    garden.doi = "10.55555/fake-doi"
    garden.validate()


def test_auto_doi(garden_no_fields):
    garden = garden_no_fields
    assert not garden.doi
    garden.authors = ["Mendel, Gregor"]
    garden.title = "Experiments on Plant Hybridization"
    garden.request_doi()
    assert garden.doi


def test_register_metadata(garden_client, garden_required_fields_only, tmp_path):
    # uses pytest builtin fixture to write to tmp_path
    gc = garden_client
    garden = garden_required_fields_only
    gc.register_metadata(garden, tmp_path)
    assert (tmp_path / "metadata.json").exists()
    with open(tmp_path / "metadata.json", "r") as f:
        json_contents = f.read()
        assert json_contents == garden.json()


def test_step_wrapper():
    # well-annotated callables are accepted; poorly annotated callables are not
    @step
    def well_annotated(a: int, b: str, g: Garden) -> tuple[int, str, Garden]:
        pass

    assert isinstance(well_annotated, step)

    with pytest.raises(ValidationError):

        @step
        def incomplete_annotated(a: int, b: None, g) -> int:
            pass


def test_pipeline_compose_union():
    # check that pipelines can correctly compose tricky functions, which may be
    # annotated like typing.Union *or* like (A | B)
    @step
    def wants_int_or_str(arg: int | str) -> str | int:
        pass

    @step
    def wants_int_or_str_old_syntax(arg: Union[int, str]) -> Union[str, int]:
        pass

    @step
    def str_only(arg: str) -> str:
        pass

    good = Pipeline(
        authors=["mendel"],
        title="good pipeline",
        steps=[str_only, wants_int_or_str],
    )

    also_good = Pipeline(
        authors=["mendel"],
        title="good pipeline",
        steps=[str_only, wants_int_or_str_old_syntax],
    )

    union_order_doesnt_matter = Pipeline(
        authors=["mendel"],
        title="good pipeline",
        steps=[wants_int_or_str, wants_int_or_str],
    )

    union_syntax_doesnt_matter = Pipeline(
        authors=["mendel"],
        title="good pipeline",
        steps=[wants_int_or_str, wants_int_or_str_old_syntax],
    )

    with pytest.raises(ValidationError):
        union_str_does_not_subtype_str = Pipeline(
            authors=["mendel"],
            title="bad pipeline",
            steps=[wants_int_or_str, str_only],
        )

    with pytest.raises(ValidationError):
        old_union_str_does_not_subtype_str = Pipeline(
            authors=["mendel"],
            title="bad pipeline",
            steps=[wants_int_or_str_old_syntax, str_only],
        )
    return


def test_pipeline_compose_tuple():
    # check that pipelines can correctly compose tricky
    # functions which may or may not expect a plain tuple
    # to be treated as *args
    @step
    def returns_tuple(a: int, b: str) -> tuple[int, str]:
        pass

    @step
    def wants_tuple_as_tuple(t: tuple[int, str]) -> int:
        pass

    @step
    def wants_tuple_as_args(x: int, y: str) -> str:
        pass

    @step
    def wants_flipped_tuple_as_args(arg1: str, arg2: int) -> float:
        pass

    good = Pipeline(
        authors=["mendel"],
        title="good pipeline",
        steps=[returns_tuple, wants_tuple_as_tuple],
    )

    with pytest.raises(ValidationError):
        bad = Pipeline(
            authors=["mendel"],
            title="backwards pipeline",
            steps=[wants_tuple_as_tuple, returns_tuple],
        )

    ugly = Pipeline(
        authors=["mendel"],
        title="ugly (using *args) but allowed pipeline",
        steps=[returns_tuple, wants_tuple_as_args],
    )
    with pytest.raises(ValidationError):
        ugly_and_bad = Pipeline(
            authors=["mendel"],
            title="ugly (using *args) and disallowed pipeline",
            steps=[returns_tuple, wants_flipped_tuple_as_args],
        )


def test_authors_are_contributors():
    # TODO
    # verify that adding a pipeline with new authors
    # updates the garden's 'contributors' field
    pass
