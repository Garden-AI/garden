import pytest
from typer.testing import CliRunner

from garden_ai import local_data
from garden_ai.app.main import app
from garden_ai.client import GardenClient

import re

runner = CliRunner()


@pytest.mark.cli
def test_garden_create(garden_all_fields, tmp_path, mocker):
    mock_client = mocker.MagicMock(GardenClient)
    mocker.patch("garden_ai.app.garden.GardenClient").return_value = mock_client
    mocker.patch("garden_ai.app.garden.local_data.put_local_garden").return_value = None
    mock_client.create_garden.return_value = garden_all_fields

    command = [
        "garden",
        "create",
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

    # regular expression to parse out ANSI escape sequences from rich print
    assert garden_all_fields.doi in re.compile(r"\x1b[^m]*m").sub("", result.stdout)


@pytest.mark.cli
def test_garden_list(database_with_connected_pipeline, tmp_path, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_pipeline
    )

    garden_uuid = "e1a3b50b-4efc-42c8-8422-644f4f858b87"
    garden_title = "Will Test Garden"
    garden_doi = "10.23677/fake-doi"

    command = [
        "garden",
        "list",
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    assert garden_uuid in result.stdout
    assert garden_title in result.stdout
    assert garden_doi in result.stdout


@pytest.mark.cli
@pytest.mark.parametrize("use_doi", [True, False])
def test_garden_pipeline_add(database_with_unconnected_pipeline, mocker, use_doi):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_unconnected_pipeline
    )

    garden_uuid = "e1a3b50b-4efc-42c8-8422-644f4f858b87"
    pipeline_uuid = "b537520b-e86e-45bf-8566-4555a72b0b08"
    pipeline_doi = "10.23677/jx31-gx98"

    def run_test_with_ids(garden_id, pipeline_id):
        before_addition = local_data.get_local_garden_by_uuid(garden_id)
        assert len(before_addition.pipeline_ids) == 0

        command = ["garden", "add-pipeline", "-g", garden_id, "-p", pipeline_id]
        result = runner.invoke(app, command)
        assert result.exit_code == 0

        garden_after_addition = local_data.get_local_garden_by_uuid(garden_id)
        # expanded metadata includes "pipelines" attribute
        after_addition = garden_after_addition.expanded_metadata()
        assert len(after_addition["pipelines"]) == 1
        assert str(after_addition["pipelines"][0]["uuid"]) == pipeline_uuid
        assert after_addition["pipelines"][0]["doi"] == pipeline_doi

    if use_doi:
        run_test_with_ids(garden_uuid, pipeline_doi)
    else:
        run_test_with_ids(garden_uuid, pipeline_uuid)


@pytest.mark.cli
@pytest.mark.parametrize("use_doi", [True, False])
def test_garden_publish(database_with_connected_pipeline, mocker, use_doi):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_pipeline
    )
    mock_client = mocker.MagicMock(GardenClient)
    mocker.patch("garden_ai.app.garden.GardenClient").return_value = mock_client

    garden_uuid = "e1a3b50b-4efc-42c8-8422-644f4f858b87"
    garden_doi = "10.23677/fake-doi"
    pipeline_uuid = "b537520b-e86e-45bf-8566-4555a72b0b08"
    mock_client._mint_doi.return_value = garden_doi

    def run_test_with_id(garden_id):
        command = [
            "garden",
            "publish",
            "-g",
            garden_id,
        ]
        result = runner.invoke(app, command)
        assert result.exit_code == 0

        mock_client.publish_garden_metadata.assert_called_once()

        args = mock_client.publish_garden_metadata.call_args.args
        garden = args[0]
        # Confirm that expanded gardens include pipelines
        denormalized_garden_metadata = garden.expanded_metadata()
        assert denormalized_garden_metadata["pipelines"][0]["steps"] is not None
        assert (
            str(denormalized_garden_metadata["pipelines"][0]["uuid"]) == pipeline_uuid
        )
        # Confirm that pipelines within expanded gardens contain models
        model = denormalized_garden_metadata["pipelines"][0]["models"][0]
        assert model["version"] == "3"
        assert len(model["connections"]) == 1

    if use_doi:
        run_test_with_id(garden_doi)
    else:
        run_test_with_id(garden_uuid)


@pytest.mark.cli
def test_garden_pipeline_add_with_alias(database_with_connected_pipeline, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_pipeline
    )

    garden_id = "e1a3b50b-4efc-42c8-8422-644f4f858b87"
    pipeline_id = "b537520b-e86e-45bf-8566-4555a72b0b08"
    pipeline_old_name = "fixture_pipeline"
    pipeline_alias = "fixed_ur_pipeline"

    before_addition = local_data.get_local_garden_by_uuid(garden_id)
    assert len(before_addition.pipelines) == 1
    assert hasattr(before_addition, pipeline_old_name)
    assert not hasattr(before_addition, pipeline_alias)

    command = [
        "garden",
        "add-pipeline",
        "-g",
        garden_id,
        "-p",
        pipeline_id,
        "-a",
        pipeline_alias,
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    after_addition = local_data.get_local_garden_by_uuid(garden_id)
    assert not hasattr(after_addition, pipeline_old_name)
    assert hasattr(after_addition, pipeline_alias)
