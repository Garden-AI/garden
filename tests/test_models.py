import json

from garden_ai import (
    PublishedGarden,
    RegisteredEntrypoint,
    local_data,
    garden_entrypoint,
    garden_step,
    EntrypointMetadata,
)
from garden_ai.model_connectors import HFConnector, GitHubConnector
from unittest.mock import MagicMock


def test_create_empty_garden(garden_client):
    # feels silly, but we do want users to be able to initialize an empty garden
    # & fill in required stuff later

    # object should exist with default-illegal fields
    garden = garden_client.create_garden()

    assert not garden.authors
    assert not garden.title


def test_garden_datacite(garden_title_authors_doi_only):
    data = json.loads(
        PublishedGarden.from_garden(garden_title_authors_doi_only).datacite_json()
    )

    assert isinstance(data["creators"], list)
    assert isinstance(data["titles"], list)
    assert data["publisher"] == "thegardens.ai"


def test_entrypoint_datacite(registered_entrypoint_toy_example):
    data = json.loads(registered_entrypoint_toy_example.datacite_json())

    assert isinstance(data["creators"], list)
    assert isinstance(data["titles"], list)
    assert data["publisher"] == "thegardens.ai"


def test_garden_can_access_entrypoint_as_attribute(
    mocker, database_with_connected_entrypoint
):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_entrypoint
    )
    garden = local_data.get_local_garden_by_doi("10.23677/fake-doi")
    published = PublishedGarden.from_garden(garden)
    assert isinstance(published.fixture_entrypoint, RegisteredEntrypoint)


def test_garden_entrypoint_decorator():
    entrypoint_meta = EntrypointMetadata(
        title="My Entrypoint",
        authors=["Willie", "Waylon", "Johnny", "Kris"],
        description="A garden entrypoint",
        tags=["garden_ai"],
    )

    model_connector = HFConnector("willengler-uc/iris-classifier")

    @garden_entrypoint(
        metadata=entrypoint_meta,
        model_connectors=[model_connector],
        garden_doi="10.23677/fake-doi",
    )
    def my_entrypoint():
        pass

    assert my_entrypoint._garden_entrypoint.title == "My Entrypoint"
    models = my_entrypoint._garden_entrypoint.models
    assert len(models) == 1
    assert models[0].model_identifier == "willengler-uc/iris-classifier"
    assert my_entrypoint._garden_entrypoint._target_garden_doi == "10.23677/fake-doi"


def test_garden_entrypoint_decorator_github():
    entrypoint_meta = EntrypointMetadata(
        title="My Entrypoint",
        authors=["Test", "Jef"],
        description="A garden entrypoint",
        tags=["garden_ai"],
    )

    model_connector = GitHubConnector("https://github.com/uw-cmg/ASR_model")

    @garden_entrypoint(
        metadata=entrypoint_meta,
        model_connectors=[model_connector],
        garden_doi="10.23671/fake-doi",
    )
    def my_entrypoint():
        pass

    assert my_entrypoint._garden_entrypoint.title == "My Entrypoint"
    models = my_entrypoint._garden_entrypoint.models
    assert len(models) == 1
    assert models[0].model_identifier == "https://github.com/uw-cmg/ASR_model"
    assert my_entrypoint._garden_entrypoint._target_garden_doi == "10.23671/fake-doi"


def test_garden_step_decorator():
    @garden_step(description="My nifty step")
    def my_step():
        pass

    assert my_step._garden_step.function_name == "my_step"
    assert my_step._garden_step.description == "My nifty step"


def test_GHconnector_idempotent(mocker):
    # Mock os.path.exists and os.mkdir
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("os.mkdir")

    # Mock Repo to simulate both clone and pull scenarios
    mock_repo_class = mocker.patch("garden_ai.model_connectors.github_conn.Repo")
    mock_repo_instance = MagicMock()
    mock_repo_class.return_value = mock_repo_instance
    mock_repo_instance.remotes.origin.pull = MagicMock()

    # Mock Repo.clone_from method to track calls without actually cloning
    mock_clone_from = mocker.patch(
        "garden_ai.model_connectors.github_conn.Repo.clone_from"
    )

    # Mock is_git_dir to control the flow in the stage method
    mocker.patch(
        "garden_ai.model_connectors.github_conn.is_git_dir",
        side_effect=[
            False,
            True,
            True,
        ],  # First call: not a git dir, then it is a git dir
    )

    # enable_imports=False just bc mocking sys.path.append was hard
    connector = GitHubConnector(
        repo_url="https://github.com/fake/repo",
        local_dir="gh_model",
        branch="main",
        enable_imports=False,
    )

    # First call should trigger clone since it's not a git dir yet
    connector.stage()
    mock_clone_from.assert_called_once_with(
        "https://github.com/fake/repo.git", "gh_model", branch="main"
    )

    # Reset mock to test idempotency on subsequent calls
    mock_clone_from.reset_mock()

    # Subsequent calls should not trigger clone_from again, but should pull
    connector.stage()
    connector.stage()

    # Assert that Repo.clone_from was not called again after the first time
    mock_clone_from.assert_not_called()
    # Assert that pull was called on subsequent invocations
    assert (
        mock_repo_instance.remotes.origin.pull.call_count == 2
    ), "Pull should be called on subsequent calls"
