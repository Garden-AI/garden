import pytest
import sys
from garden_ai.app.main import app
from garden_ai.client import GardenClient
from typer.testing import CliRunner
import string
import random
from keyword import iskeyword
from garden_ai.app.pipeline import validate_identifier

runner = CliRunner()


@pytest.mark.cli
@pytest.mark.skipif(
    sys.version_info < (3, 8), reason="mocked call_args.kwargs breaks pre-3.8"
)
def test_garden_create(garden_all_fields, tmp_path, mocker):
    mock_client = mocker.MagicMock(GardenClient)
    mocker.patch("garden_ai.app.garden.GardenClient").return_value = mock_client

    command = [
        "garden",
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
    mock_client.put_local.assert_called_once()


@pytest.mark.skipif(
    sys.version_info < (3, 8), reason="mocked call_args.kwargs breaks pre-3.8"
)
def test_pipeline_create(pipeline_toy_example, mocker, tmp_path):
    mock_client = mocker.MagicMock(GardenClient)
    mocker.patch("garden_ai.app.pipeline.GardenClient").return_value = mock_client
    command = [
        "pipeline",
        "create",
        "--directory",
        str(tmp_path),
        "--title",
        pipeline_toy_example.title,
        "--description",
        pipeline_toy_example.description,
        "--year",
        pipeline_toy_example.year,
    ]
    for name in pipeline_toy_example.authors:
        command += ["--author", name]
    for name in pipeline_toy_example.contributors:
        command += ["--contributor", name]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    kwargs = mock_client.create_pipeline.call_args.kwargs
    del kwargs["steps"]  # different steps on purpose -- can't get a function from cli
    for key in kwargs:
        assert kwargs[key] == getattr(pipeline_toy_example, key)

    mock_client.put_local.assert_called_once()


def test_validate_identifier():
    possible_name = "".join(random.choices(string.printable, k=50))
    valid_name = validate_identifier(possible_name)
    assert valid_name.isidentifier()
    assert not iskeyword(validate_identifier("import"))
