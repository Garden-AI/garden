import pytest
import sys
from garden_ai.app.main import app
from garden_ai.client import GardenClient
from typer.testing import CliRunner

runner = CliRunner()


@pytest.mark.cli
@pytest.mark.skipif(sys.version_info < (3, 8), reason="can't inspect call args by name with 3.7")
def test_garden_create(garden_all_fields, tmp_path, mocker):
    mock_client = mocker.MagicMock(GardenClient)
    mocker.patch("garden_ai.app.create.GardenClient").return_value = mock_client

    command = [
        "create",
        str(tmp_path / "pea_directory"),
        "--title",
        garden_all_fields.title,
        "--description",
        garden_all_fields.description,
        "--year",
        garden_all_fields.year,
    ]
    for name in garden_all_fields.authors:
        command += ["--author", name]
    for name in garden_all_fields.contributors:
        command += ["--contributor", name]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    mock_client.create_garden.assert_called_once()
    kwargs = mock_client.create_garden.call_args.kwargs
    for key in kwargs:
        assert kwargs[key] == getattr(garden_all_fields, key)
    mock_client.register_metadata.assert_called_once()
