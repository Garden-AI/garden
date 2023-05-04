from dataclasses import dataclass
import pytest

from globus_sdk import SearchClient
from garden_ai.globus_search import garden_search


@dataclass
class MockSearchResult:
    text: str


def test_get_by_doi_valid(mocker, valid_search_by_doi):
    mock_search_client = mocker.MagicMock(SearchClient)
    mock_search_client.search = mocker.Mock(
        return_value=MockSearchResult(valid_search_by_doi)
    )
    garden = garden_search.get_remote_garden_by_doi("foo", {}, mock_search_client)
    # Simple check that garden was populated with fields
    assert garden.year == "2023"


def test_get_by_uuid_valid(mocker, valid_search_by_subject):
    mock_search_client = mocker.MagicMock(SearchClient)
    mock_search_client.get_subject = mocker.Mock(
        return_value=MockSearchResult(valid_search_by_subject)
    )
    garden = garden_search.get_remote_garden_by_uuid("foo", {}, mock_search_client)
    # Simple check that garden was populated with fields
    assert garden.year == "2023"


def test_get_by_doi_none_found(mocker, empty_search_by_doi):
    mock_search_client = mocker.MagicMock(SearchClient)
    mock_search_client.search = mocker.Mock(
        return_value=MockSearchResult(empty_search_by_doi)
    )
    with pytest.raises(garden_search.RemoteGardenException):
        garden_search.get_remote_garden_by_doi("foo", {}, mock_search_client)
