import pytest


@pytest.mark.cli
def test_shows_help_when_no_args(
    cli_runner,
    app,
):
    cli_args = [
        "entrypoint",
    ]

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    assert "Usage:" in result.output


@pytest.mark.cli
def test_add_repository_prompts_for_contributors(
    cli_runner,
    app,
    faker,
    mocker,
):
    name = faker.name()
    cli_args = [
        "entrypoint",
        "add-repository",
        "some/doi",
        "--url=http://github.com/Garden-AI/garden",
        f"--name='{name}'",
        # no contributors
    ]

    mocker.patch(
        "garden_ai.backend_client.BackendClient.get_entrypoint_metadata",
        return_value=mocker.MagicMock(),
    )
    mocker.patch("garden_ai.backend_client.BackendClient.put_entrypoint_metadata")
    mock_prompt = mocker.patch(
        "garden_ai.app.entrypoint.Prompt.ask", side_effect=["Some Contributor", None]
    )

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    mock_prompt.assert_called()
    assert name in result.output


@pytest.mark.cli
def test_add_repository_prompts_for_missing_url(
    cli_runner,
    app,
    faker,
    mocker,
):
    cli_args = [
        "entrypoint",
        "add-repository",
        "some/doi",
        # missing url
        "--name='Some Name'",
        "--contributor='Some Contributor'",
    ]

    mocker.patch(
        "garden_ai.backend_client.BackendClient.get_entrypoint_metadata",
        return_value=mocker.MagicMock(),
    )
    mocker.patch("garden_ai.backend_client.BackendClient.put_entrypoint_metadata")

    result = cli_runner.invoke(app, cli_args, input=f"{faker.url()}\n")
    assert result.exit_code == 0
    assert "The url linking to the repository" in result.output
    assert "Some Name" in result.output


@pytest.mark.cli
def test_add_repository_prompts_for_missing_respository_name(
    cli_runner,
    app,
    faker,
    mocker,
):
    url = faker.url()
    cli_args = [
        "entrypoint",
        "add-repository",
        "some/doi",
        f"--url={url}",
        # missing repository name
        f"--contributor='{faker.name()}'",
    ]

    mocker.patch(
        "garden_ai.backend_client.BackendClient.get_entrypoint_metadata",
        return_value=mocker.MagicMock(),
    )
    mocker.patch("garden_ai.backend_client.BackendClient.put_entrypoint_metadata")

    repo_name = faker.name()
    result = cli_runner.invoke(app, cli_args, input=f"{repo_name}")
    assert result.exit_code == 0
    assert "The name to display for your repository" in result.output
    assert repo_name in result.output


@pytest.mark.cli
def test_add_paper_prints_usage_when_no_args(
    cli_runner,
    app,
):
    cli_args = [
        "entrypoint",
        "add-paper",
    ]

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    assert "Usage:" in result.output


@pytest.mark.cli
def test_add_paper_requires_doi(
    cli_runner,
    app,
):
    cli_args = [
        "entrypoint",
        "add-paper",
        # missing doi
        "--title='Fake Title'",
        "--author='Some Author'",
        "--doi='some/doi'",
        "--citation='A fake citation'",
    ]

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 2
    assert "Missing argument 'ENTRYPOINT_DOI'" in result.output


@pytest.mark.cli
def test_add_paper_prompts_for_authors_if_none(
    cli_runner,
    app,
    mocker,
):
    cli_args = [
        "entrypoint",
        "add-paper",
        "some/doi",
        "--title='Some Title",
        # missing authors
        "--doi='some/doi",
        "--citation='some citation",
    ]

    mocker.patch(
        "garden_ai.backend_client.BackendClient.get_entrypoint_metadata",
        return_value=mocker.MagicMock(),
    )
    mocker.patch("garden_ai.backend_client.BackendClient.put_entrypoint_metadata")
    mock_prompt = mocker.patch(
        "garden_ai.app.entrypoint.Prompt.ask", side_effect=["Some Author", None]
    )

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    mock_prompt.assert_called()


@pytest.mark.cli
def test_add_paper_prompts_for_paper_doi_if_missing(
    cli_runner,
    app,
    mocker,
):
    cli_args = [
        "entrypoint",
        "add-paper",
        "some/doi",
        "--title='Some Title",
        "--author='Some Author'",
        # missing paper doi
        "--citation='some citation",
    ]

    mocker.patch(
        "garden_ai.backend_client.BackendClient.get_entrypoint_metadata",
        return_value=mocker.MagicMock(),
    )
    mocker.patch("garden_ai.backend_client.BackendClient.put_entrypoint_metadata")
    mock_prompt = mocker.patch(
        "garden_ai.app.entrypoint.Prompt.ask", side_effect=["some/doi", None]
    )

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    mock_prompt.assert_called()


@pytest.mark.cli
def test_add_paper_prompts_for_citation_if_missing(
    cli_runner,
    app,
    mocker,
):
    cli_args = [
        "entrypoint",
        "add-paper",
        "some/doi",
        "--title='Some Title",
        "--author='Some Author'",
        "--doi='some/doi'",
        # missing citation
    ]

    mocker.patch(
        "garden_ai.backend_client.BackendClient.get_entrypoint_metadata",
        return_value=mocker.MagicMock(),
    )
    mocker.patch("garden_ai.backend_client.BackendClient.put_entrypoint_metadata")
    mock_prompt = mocker.patch(
        "garden_ai.app.entrypoint.Prompt.ask", side_effect=["Some citation", None]
    )

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    mock_prompt.assert_called()


@pytest.mark.cli
def test_register_doi_prints_usage_when_no_args(
    cli_runner,
    app,
):
    cli_args = [
        "entrypoint",
        "register-doi",
    ]

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    assert "Usage:" in result.output


@pytest.mark.cli
def test_register_doi_prints_doi_on_success(
    cli_runner,
    app,
    mocker,
):
    cli_args = [
        "entrypoint",
        "register-doi",
        "some/doi",
    ]

    mocker.patch("garden_ai.client.GardenClient.register_entrypoint_doi")

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    assert "some/doi" in result.output


@pytest.mark.cli
def test_show_prints_entrypoint_json(
    cli_runner,
    app,
    mocker,
):
    cli_args = [
        "entrypoint",
        "show",
        "some/doi",
        "some/other.doi",
    ]

    mocker.patch(
        "garden_ai.backend_client.BackendClient.get_entrypoints",
        return_value=[mocker.MagicMock()],
    )
    mock_rich = mocker.patch(
        "garden_ai.app.entrypoint.rich", return_value=mocker.MagicMock()
    )

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    mock_rich.print_json.assert_called()


@pytest.mark.cli
def test_edit_requires_doi(
    cli_runner,
    app,
):
    cli_args = [
        "entrypoint",
        "edit",
    ]

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 2
    assert "Missing argument 'DOI'" in result.output


@pytest.mark.cli
def test_edit_prints_doi_on_successful_edit(
    cli_runner,
    app,
    mocker,
):
    cli_args = [
        "entrypoint",
        "edit",
        "some/doi",
    ]

    mocker.patch(
        "garden_ai.backend_client.BackendClient.get_entrypoint_metadata",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "garden_ai.app.entrypoint.gui_edit_garden_entity",
        return_value=mocker.MagicMock(),
    )
    mocker.patch("garden_ai.backend_client.BackendClient.put_entrypoint_metadata")

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    assert "some/doi" in result.output
