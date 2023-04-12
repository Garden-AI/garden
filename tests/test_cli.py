import json

import pytest
from garden_ai.app.main import app
from garden_ai.client import GardenClient
from typer.testing import CliRunner
import string
import random
from keyword import iskeyword
from garden_ai.app.pipeline import clean_identifier
from garden_ai import local_data

runner = CliRunner()


@pytest.mark.cli
def test_garden_pipeline_add(database_with_unconnected_pipeline, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_unconnected_pipeline
    )
    garden_id = "e1a3b50b-4efc-42c8-8422-644f4f858b87"
    pipeline_id = "b537520b-e86e-45bf-8566-4555a72b0b08"

    def get_garden_meta():
        return json.loads(str(local_data.get_local_garden(garden_id)))

    before_addition = get_garden_meta()
    assert len(before_addition["pipelines"]) == 0

    command = ["garden", "add-pipeline", "-g", garden_id, "-p", pipeline_id]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    after_addition = get_garden_meta()
    assert len(after_addition["pipelines"]) == 1
    assert after_addition["pipelines"][0]["uuid"] == pipeline_id
    assert after_addition["pipelines"][0]["doi"] == "10.23677/jx31-gx98"


@pytest.mark.cli
def test_garden_publish(database_with_connected_pipeline, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_pipeline
    )
    mock_client = mocker.MagicMock(GardenClient)
    mocker.patch("garden_ai.app.garden.GardenClient").return_value = mock_client

    garden_id = "e1a3b50b-4efc-42c8-8422-644f4f858b87"
    pipeline_id = "b537520b-e86e-45bf-8566-4555a72b0b08"

    command = [
        "garden",
        "publish",
        "-g",
        garden_id,
    ]
    # We want to check that we called client.publish with the right shit
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    args = mock_client.publish_garden_metadata.call_args.args
    denormalized_garden_metadata = args[0]
    assert denormalized_garden_metadata["pipelines"][0]["steps"] is not None
    assert denormalized_garden_metadata["pipelines"][0]["uuid"] == pipeline_id


@pytest.mark.cli
def test_garden_create(garden_all_fields, tmp_path, mocker):
    mock_client = mocker.MagicMock(GardenClient)
    mocker.patch("garden_ai.app.garden.GardenClient").return_value = mock_client
    mocker.patch("garden_ai.app.garden.local_data.put_local_garden").return_value = None

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
def test_model_upload(mocker, tmp_path):
    mock_client = mocker.MagicMock(GardenClient)
    mocker.patch("garden_ai.app.model.GardenClient").return_value = mock_client
    command = [
        "model",
        "register",
        "unit-test-model",
        str(tmp_path),
        "sklearn",
        "--extra-pip-requirements",
        "torch==1.13.1",
        "--extra-pip-requirements",
        "pandas<=1.5.0",
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    args = mock_client.log_model.call_args.args
    assert args[0] == str(tmp_path)
    assert args[1] == "unit-test-model"
    assert args[2] == "sklearn"
    assert args[3] == ["torch==1.13.1", "pandas<=1.5.0"]


def test_clean_identifier():
    possible_name = "".join(random.choices(string.printable, k=50))
    valid_name = clean_identifier(possible_name)
    assert valid_name.isidentifier()
    assert not iskeyword(clean_identifier("import"))
