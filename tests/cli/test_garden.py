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


def test_garden_list(database_with_connected_entrypoint, tmp_path, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_entrypoint
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
def test_garden_show(database_with_connected_entrypoint, tmp_path, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_entrypoint
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
def test_garden_entrypoint_add(database_with_unconnected_entrypoint, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_unconnected_entrypoint
    )

    garden_doi = "10.23677/fake-doi"
    entrypoint_doi = "10.23677/jx31-gx98"

    before_addition = local_data.get_local_garden_by_doi(garden_doi)
    assert len(before_addition.entrypoint_ids) == 0

    command = ["garden", "add-entrypoint", "-g", garden_doi, "-p", entrypoint_doi]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    garden_after_addition = local_data.get_local_garden_by_doi(garden_doi)
    # expanded metadata includes "entrypoints" attribute
    after_addition = garden_after_addition.expanded_metadata()
    assert len(after_addition["entrypoints"]) == 1
    assert after_addition["entrypoints"][0]["doi"] == entrypoint_doi


@pytest.mark.cli
def test_garden_publish(database_with_connected_entrypoint, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_entrypoint
    )
    mock_client = mocker.MagicMock(GardenClient)
    mocker.patch("garden_ai.app.garden.GardenClient").return_value = mock_client

    garden_doi = "10.23677/fake-doi"
    entrypoint_doi = "10.23677/jx31-gx98"
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
    # Confirm that expanded gardens include entrypoints
    denormalized_garden_metadata = garden.expanded_metadata()
    assert str(denormalized_garden_metadata["entrypoints"][0]["doi"]) == entrypoint_doi


@pytest.mark.cli
def test_register_doi(database_with_connected_entrypoint, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_entrypoint
    )
    mock_client = mocker.MagicMock(GardenClient)
    mocker.patch("garden_ai.app.garden.GardenClient").return_value = mock_client

    garden_doi = "10.23677/fake-doi"
    entrypoint_doi = "10.23677/jx31-gx98"
    mock_client._mint_draft_doi.return_value = garden_doi

    command = [
        "garden",
        "register-doi",
        garden_doi,
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    mock_client.publish_garden_metadata.assert_called_once()

    args = mock_client.publish_garden_metadata.call_args.args
    kwargs = mock_client.publish_garden_metadata.call_args.kwargs
    garden = args[0]
    should_register_doi = kwargs["register_doi"]
    # Confirm that expanded gardens include entrypoints
    denormalized_garden_metadata = garden.expanded_metadata()
    assert should_register_doi
    assert str(denormalized_garden_metadata["entrypoints"][0]["doi"]) == entrypoint_doi


@pytest.mark.cli
def test_garden_entrypoint_add_with_alias(database_with_connected_entrypoint, mocker):
    def get_names(garden):
        return [
            garden.entrypoint_aliases.get(cached.doi) or cached.short_name
            for cached in garden._entrypoint_cache
        ]

    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_entrypoint
    )

    garden_doi = "10.23677/fake-doi"
    entrypoint_doi = "10.23677/jx31-gx98"
    entrypoint_old_name = "fixture_entrypoint"
    entrypoint_alias = "fixed_ur_entrypoint"

    before_addition = local_data.get_local_garden_by_doi(garden_doi)
    entrypoint_names = get_names(before_addition)

    assert len(before_addition.entrypoint_ids) == 1
    assert entrypoint_old_name in entrypoint_names
    assert entrypoint_alias not in entrypoint_names

    command = [
        "garden",
        "add-entrypoint",
        "-g",
        garden_doi,
        "-p",
        entrypoint_doi,
        "-a",
        entrypoint_alias,
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    after_addition = local_data.get_local_garden_by_doi(garden_doi)
    new_entrypoint_names = get_names(after_addition)

    assert entrypoint_old_name not in new_entrypoint_names
    assert entrypoint_alias in new_entrypoint_names


@pytest.mark.cli
def test_delete_garden_exists(mocker, database_with_connected_entrypoint):
    mock_client = mocker.MagicMock(GardenClient)
    mocker.patch("garden_ai.app.garden.GardenClient").return_value = mock_client

    mocker.patch("garden_ai.client.GardenClient.delete_garden")
    mocker.patch("typer.confirm", return_value=True)
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_entrypoint
    )

    garden_doi = "10.23677/fake-doi"
    command = ["garden", "delete", "-g", garden_doi]
    result = runner.invoke(app, command)

    assert result.exit_code == 0


@pytest.mark.cli
def test_delete_garden_not_exists_override(mocker, database_with_connected_entrypoint):
    mock_client = mocker.MagicMock(GardenClient)
    mocker.patch("garden_ai.app.garden.GardenClient").return_value = mock_client

    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_entrypoint
    )
    mocker.patch("garden_ai.client.GardenClient.delete_garden")
    mocker.patch("typer.confirm", return_value=True)

    garden_doi = "10.23677/nonexistent-doi"
    command = ["garden", "delete", "-g", garden_doi, "--dangerous-override"]
    result = runner.invoke(app, command)

    assert result.exit_code == 0
