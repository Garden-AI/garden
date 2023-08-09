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


def test_garden_list(database_with_connected_pipeline, tmp_path, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_pipeline
    )

    garden_title = "Will Test Garden"
    garden_doi = "10.23677/fake-doi"

    command = [
        "garden",
        "list",
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    assert garden_title in result.stdout
    assert garden_doi in result.stdout


@pytest.mark.cli
def test_garden_show(database_with_connected_pipeline, tmp_path, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_pipeline
    )

    garden_title = "Will Test Garden"
    garden_doi = "10.23677/fake-doi"

    command = [
        "garden",
        "show",
        garden_doi,
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    assert garden_title in result.stdout
    assert garden_doi in result.stdout

    command = [
        "garden",
        "show",
        "not_a_garden_id",
        garden_doi,
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    assert garden_title in result.stdout
    assert garden_doi in result.stdout


@pytest.mark.cli
def test_garden_pipeline_add(database_with_unconnected_pipeline, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_unconnected_pipeline
    )

    garden_doi = "10.23677/fake-doi"
    pipeline_doi = "10.23677/jx31-gx98"

    before_addition = local_data.get_local_garden_by_doi(garden_doi)
    assert len(before_addition.pipeline_ids) == 0

    command = ["garden", "add-pipeline", "-g", garden_doi, "-p", pipeline_doi]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    garden_after_addition = local_data.get_local_garden_by_doi(garden_doi)
    # expanded metadata includes "pipelines" attribute
    after_addition = garden_after_addition.expanded_metadata()
    assert len(after_addition["pipelines"]) == 1
    assert after_addition["pipelines"][0]["doi"] == pipeline_doi


@pytest.mark.cli
def test_garden_publish(database_with_connected_pipeline, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_pipeline
    )
    mock_client = mocker.MagicMock(GardenClient)
    mocker.patch("garden_ai.app.garden.GardenClient").return_value = mock_client

    garden_doi = "10.23677/fake-doi"
    pipeline_doi = "10.23677/jx31-gx98"
    mock_client._mint_draft_doi.return_value = garden_doi

    command = [
        "garden",
        "publish",
        "-g",
        garden_doi,
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    mock_client.publish_garden_metadata.assert_called_once()

    args = mock_client.publish_garden_metadata.call_args.args
    garden = args[0]
    # Confirm that expanded gardens include pipelines
    denormalized_garden_metadata = garden.expanded_metadata()
    assert denormalized_garden_metadata["pipelines"][0]["steps"] is not None
    assert str(denormalized_garden_metadata["pipelines"][0]["doi"]) == pipeline_doi


@pytest.mark.cli
def test_garden_pipeline_add_with_alias(database_with_connected_pipeline, mocker):
    def get_names(garden):
        return (
            garden.pipeline_aliases.get(cached.short_name) or cached.short_name
            for cached in garden._pipeline_cache
        )

    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_pipeline
    )

    garden_doi = "10.23677/fake-doi"
    pipeline_doi = "10.23677/jx31-gx98"
    pipeline_old_name = "fixture_pipeline"
    pipeline_alias = "fixed_ur_pipeline"

    before_addition = local_data.get_local_garden_by_doi(garden_doi)
    pipeline_names = get_names(before_addition)

    assert len(before_addition.pipeline_ids) == 1
    assert pipeline_old_name in pipeline_names
    assert pipeline_alias not in pipeline_names

    command = [
        "garden",
        "add-pipeline",
        "-g",
        garden_doi,
        "-p",
        pipeline_doi,
        "-a",
        pipeline_alias,
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    after_addition = local_data.get_local_garden_by_doi(garden_doi)
    new_pipeline_names = get_names(after_addition)

    assert pipeline_old_name not in new_pipeline_names
    assert pipeline_alias in new_pipeline_names
