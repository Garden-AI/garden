# flake8: noqa: F841
import re

import pytest
import rich

from garden_ai.client import GardenClient
from garden_ai.gardens import Garden


@pytest.mark.cli
def test_garden_no_args_prints_usage(cli_runner, app):
    result = cli_runner.invoke(app, ["garden"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout


@pytest.mark.cli
def test_garden_create_all_args_succeeds(
    cli_runner,
    app,
    patch_backend_client_requests,
    mock_GardenMetadata,
    mocker,
):
    garden = mock_GardenMetadata
    cli_args = [
        "garden",
        "create",
        f"--title={garden.title}",
        f"--author={garden.authors[0]}",
        f"--year={garden.year}",
        f"--contributor={garden.contributors[0]}",
        f"--contributor={garden.contributors[1]}",
        f"--description={garden.description}",
        f"--tag={garden.tags[0]}",
        f"--tag={garden.tags[1]}",
    ]

    mocker.patch(
        "garden_ai.client.GardenClient._mint_draft_doi",
        return_value=garden.doi,
    )
    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    assert garden.doi in result.stdout


@pytest.mark.cli
def test_garden_create_prompts_for_missing_title(
    cli_runner,
    app,
    patch_backend_client_requests,
    mock_GardenMetadata,
    mocker,
):
    garden = mock_GardenMetadata
    cli_args = [
        "garden",
        "create",
        # missing title
        f"--author={garden.authors[0]}",
        f"--year={garden.year}",
        f"--contributor={garden.contributors[0]}",
        f"--description={garden.description}",
        f"--tag={garden.tags[0]}",
        f"--tag={garden.tags[1]}",
    ]

    mocker.patch(
        "garden_ai.client.GardenClient._mint_draft_doi",
        return_value=garden.doi,
    )

    result = cli_runner.invoke(app, cli_args, input="Some Title\n")
    assert result.exit_code == 0
    assert "Please enter a title" in result.stdout


@pytest.mark.cli
def test_garden_create_prompts_for_missing_author(
    cli_runner,
    app,
    patch_backend_client_requests,
    mock_GardenMetadata,
    mocker,
):
    garden = mock_GardenMetadata
    cli_args = [
        "garden",
        "create",
        f"--title={garden.title}",
        # missing author
        f"--year={garden.year}",
        f"--contributor={garden.contributors[0]}",
        f"--description={garden.description}",
        f"--tag={garden.tags[0]}",
        f"--tag={garden.tags[1]}",
    ]

    mocker.patch(
        "garden_ai.client.GardenClient._mint_draft_doi",
        return_value=garden.doi,
    )

    mock_prompt = mocker.patch(
        "garden_ai.app.garden.Prompt.ask",
        side_effect=["SomeAuthor", None],
    )

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    mock_prompt.assert_called()
    assert garden.doi in result.stdout


@pytest.mark.cli
def test_garden_create_prompts_for_missing_contributor(
    cli_runner,
    app,
    patch_backend_client_requests,
    mock_GardenMetadata,
    mocker,
):
    garden = mock_GardenMetadata
    cli_args = [
        "garden",
        "create",
        f"--title={garden.title}",
        f"--author={garden.authors[0]}",
        f"--year={garden.year}",
        # missing contributor
        f"--description={garden.description}",
        f"--tag={garden.tags[0]}",
        f"--tag={garden.tags[1]}",
    ]

    mocker.patch(
        "garden_ai.client.GardenClient._mint_draft_doi",
        return_value=garden.doi,
    )
    mock_prompt = mocker.patch(
        "garden_ai.app.garden.Prompt.ask",
        side_effect=["SomeContributor", None],
    )

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    mock_prompt.assert_called()
    assert garden.doi in result.stdout


@pytest.mark.cli
def test_garden_create_prompts_for_missing_description(
    cli_runner,
    app,
    patch_backend_client_requests,
    mock_GardenMetadata,
    mocker,
):
    garden = mock_GardenMetadata
    cli_args = [
        "garden",
        "create",
        f"--title={garden.title}",
        f"--author={garden.authors[0]}",
        f"--year={garden.year}",
        f"--contributor={garden.contributors[0]}",
        # missing description
        f"--tag={garden.tags[0]}",
        f"--tag={garden.tags[1]}",
    ]

    mocker.patch(
        "garden_ai.client.GardenClient._mint_draft_doi",
        return_value=garden.doi,
    )

    mock_prompt = mocker.patch(
        "garden_ai.app.garden.Prompt.ask",
        side_effect=["SomeDescription", None],
    )

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    mock_prompt.assert_called()
    assert garden.doi in result.stdout


@pytest.mark.cli
def test_add_entrypoint_valid_args(
    cli_runner,
    app,
    garden_nested_metadata_json,
    mock_RegisteredEntrypointMetadata,
    mock_GardenMetadata,
    mocker,
):
    mock_get_garden_metadata = mocker.patch(
        "garden_ai.backend_client.BackendClient.get_garden_metadata",
        return_value=mock_GardenMetadata,
    )

    mock_get_entrypoint_metadata = mocker.patch(
        "garden_ai.backend_client.BackendClient.get_entrypoint_metadata",
        return_value=mock_RegisteredEntrypointMetadata,
    )

    mock_put_garden = mocker.patch(
        "garden_ai.backend_client.BackendClient.put_garden",
        return_value=garden_nested_metadata_json,
    )

    cli_args = [
        "garden",
        "add-entrypoint",
        f"--entrypoint={mock_RegisteredEntrypointMetadata.doi}",
        f"--garden={mock_GardenMetadata.doi}",
        "--alias=my_entrypoint",
    ]

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    assert result.exception is None
    assert mock_RegisteredEntrypointMetadata.doi in result.stdout


@pytest.mark.cli
def test_add_entrypoint_rejects_invalid_alias(
    cli_runner,
    app,
    garden_nested_metadata_json,
    mock_RegisteredEntrypointMetadata,
    mock_GardenMetadata,
    mocker,
):
    mock_get_garden_metadata = mocker.patch(
        "garden_ai.backend_client.BackendClient.get_garden_metadata",
        return_value=mock_GardenMetadata,
    )

    mock_get_entrypoint_metadata = mocker.patch(
        "garden_ai.backend_client.BackendClient.get_entrypoint_metadata",
        return_value=mock_RegisteredEntrypointMetadata,
    )

    mock_put_garden = mocker.patch(
        "garden_ai.backend_client.BackendClient.put_garden",
        return_value=garden_nested_metadata_json,
    )

    cli_args = [
        "garden",
        "add-entrypoint",
        f"--entrypoint={mock_RegisteredEntrypointMetadata.doi}",
        f"--garden={mock_GardenMetadata.doi}",
        "--alias='not a valid identifier",
    ]

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 1
    assert result.exception is not None
    assert "alias must be a valid python identifier" in str(result.exception)


@pytest.mark.cli
def test_delete_prompts_for_confirmation(
    cli_runner,
    app,
    mocker,
):
    cli_args = [
        "garden",
        "delete",
        "some/doi",
    ]
    mocker.patch("garden_ai.backend_client.BackendClient.delete_garden")
    result = cli_runner.invoke(app, cli_args, input="Y")
    assert result.exit_code == 0
    assert "Are you sure you want to proceed?" in result.stdout
    assert "Garden some/doi has been deleted" in result.stdout


@pytest.mark.cli
def test_delete_aborts_on_failed_confirmation(
    cli_runner,
    app,
    mocker,
):
    cli_args = [
        "garden",
        "delete",
        "some/doi",
    ]
    mocker.patch("garden_ai.backend_client.BackendClient.delete_garden")
    result = cli_runner.invoke(app, cli_args, input="N")
    assert result.exit_code == 1
    assert "Are you sure you want to proceed?" in result.stdout
    assert "Garden some/doi has been deleted" not in result.stdout


@pytest.mark.cli
def test_register_doi(
    cli_runner,
    app,
    mocker,
):
    cli_args = [
        "garden",
        "register-doi",
        "some/doi",
    ]
    mocker.patch("garden_ai.client.GardenClient.register_garden_doi")
    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    assert "some/doi has been moved out of draft status" in result.stdout


@pytest.mark.cli
def test_list_displays_correctly(
    cli_runner,
    app,
    garden_nested_metadata_json,
    mocker,
):
    cli_args = [
        "garden",
        "list",
    ]

    mocker.patch(
        "garden_ai.client.GardenClient.get_user_identity_id", return_value="some-uuid"
    )
    mocker.patch(
        "garden_ai.backend_client.BackendClient.get_gardens",
        return_value=[
            Garden._from_nested_metadata(garden_nested_metadata_json) for _ in range(5)
        ],
    )

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0

    dois = re.findall(rf"{garden_nested_metadata_json['doi']}", result.stdout)
    assert len(dois) == 5


@pytest.mark.cli
def test_show_displays_garden_json(
    cli_runner,
    app,
    garden_nested_metadata_json,
    mocker,
):
    cli_args = [
        "garden",
        "show",
    ]
    dois = [f"{garden_nested_metadata_json['doi']}" for _ in range(5)]
    cli_args.extend(dois)

    # Create a mock return value for get_gardens
    mock_gardens = [
        Garden._from_nested_metadata(garden_nested_metadata_json) for _ in range(5)
    ]

    # Mock the get_gardens method
    mocker.patch(
        "garden_ai.backend_client.BackendClient.get_gardens",
        return_value=mock_gardens,
    )

    mock_rich = mocker.patch(
        "garden_ai.app.garden.rich", return_value=mocker.MagicMock()
    )
    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    mock_rich.print_json.assert_called_with(
        json=mock_gardens[0].metadata.model_dump_json()
    )


@pytest.mark.cli
def test_edit_returns_failure_status_if_no_garden(
    cli_runner, app, garden_nested_metadata_json, mocker
):
    cli_args = [
        "garden",
        "edit",
        f"{garden_nested_metadata_json['doi']}",
    ]

    mocker.patch(
        "garden_ai.backend_client.BackendClient.get_garden_metadata",
        side_effect=None,
    )
    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 1
