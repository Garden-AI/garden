import string
import random
from keyword import iskeyword

import pytest
from typer.testing import CliRunner

from garden_ai.app.main import app
from garden_ai.utils.misc import clean_identifier
from garden_ai.client import GardenClient

runner = CliRunner()


@pytest.mark.cli
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


@pytest.mark.cli
def test_pipeline_list(database_with_connected_pipeline, tmp_path, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_pipeline
    )

    pipeline_uuid = "b537520b-e86e-45bf-8566-4555a72b0b08"
    pipeline_title = "Fixture pipeline"
    pipeline_doi = "10.23677/jx31-gx98"

    command = [
        "pipeline",
        "list",
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    assert pipeline_uuid in result.stdout
    assert pipeline_title in result.stdout
    assert pipeline_doi in result.stdout


@pytest.mark.cli
def test_pipeline_show(database_with_connected_pipeline, tmp_path, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_pipeline
    )

    pipeline_uuid = "b537520b-e86e-45bf-8566-4555a72b0b08"
    pipeline_title = "Fixture pipeline"
    pipeline_doi = "10.23677/jx31-gx98"

    command = [
        "pipeline",
        "show",
        pipeline_uuid,
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    assert pipeline_uuid in result.stdout
    assert pipeline_title in result.stdout
    assert pipeline_doi in result.stdout

    command = [
        "pipeline",
        "show",
        pipeline_doi,
        pipeline_uuid,
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    assert pipeline_uuid in result.stdout
    assert pipeline_title in result.stdout
    assert pipeline_doi in result.stdout


def test_clean_identifier():
    possible_name = "".join(random.choices(string.printable, k=50))
    valid_name = clean_identifier(possible_name)
    assert valid_name.isidentifier()
    assert not iskeyword(clean_identifier("import"))
