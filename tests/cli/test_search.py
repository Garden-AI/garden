from garden_ai.client import GardenClient

import pytest
import rich
from typer.testing import CliRunner

from garden_ai.app.main import app

runner = CliRunner()


@pytest.mark.cli
def test_search_easy_query(mocker):
    mock_client = mocker.MagicMock(GardenClient)
    mocker.patch("garden_ai.app.garden.GardenClient").return_value = mock_client
    mock_rich = mocker.MagicMock(rich)
    mocker.patch("garden_ai.app.garden.rich", new=mock_rich)
    command = [
        "garden",
        "search",
        "-d",
        "foo",
        "-t",
        "bar",
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    mock_client.search.assert_called_once()
    mock_rich.print_json.assert_called_once()

    args = mock_client.search.call_args.args
    query = args[0]
    assert '(title: "bar")' in query
    assert '(description: "foo")' in query
    assert " AND " in query


@pytest.mark.cli
def test_search_raw_query(mocker):
    mock_client = mocker.MagicMock(GardenClient)
    mocker.patch("garden_ai.app.garden.GardenClient").return_value = mock_client
    mock_rich = mocker.MagicMock(rich)
    mocker.patch("garden_ai.app.garden.rich", new=mock_rich)
    command = [
        "garden",
        "search",
        "-d",
        "foo",
        "--raw-query",
        "lalalala",
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    mock_client.search.assert_called_once()
    mock_rich.print_json.assert_called_once()

    args = mock_client.search.call_args.args
    query = args[0]
    assert query == "lalalala"
