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
    # Set up the mock to return a URL that matches the connector's repo_url
    mock_repo_instance.remotes.origin.url = "https://github.com/fake/repo.git"

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


def test_GitHubConnector_pins_commit_hash_correctly(mocker):
    # Mock the response from the GitHub API
    mock_response = mocker.patch("requests.get")
    mock_response.return_value.json.return_value = {"sha": "fakecommithash"}

    # Mock the Repo
    repo_url = "https://github.com/fake/repo"
    mock_repo = MagicMock()
    mock_repo.remotes.origin.url = repo_url + ".git"
    mock_repo.remotes.origin.pull = MagicMock()
    mock_repo.git.checkout = MagicMock()
    mocker.patch("garden_ai.model_connectors.github_conn.Repo", return_value=mock_repo)

    # Crete the connector
    connector = GitHubConnector(
        repo_url=repo_url,
    )

    # The revision field should be the commit hash from the latest commit
    assert connector.revision == "fakecommithash"

    # The commit hash should be stored in the metadata object as well
    assert connector.metadata.model_version == "fakecommithash"

    # stage should checkout the pinned commit when called
    connector.stage()
    mock_repo.git.checkout.assert_called_once_with("fakecommithash")


def test_HFConnector_pins_commit_hash_correctly(mocker):
    mocker.patch("os.path.exists", return_value=True)
    mock_snapshot_download = mocker.patch(
        "garden_ai.model_connectors.hugging_face.hfh.snapshot_download"
    )

    # Mock a couple of hfh.GitRefInfo objects, these are returned from hfh.list_repo_refs
    mock_main_branch = MagicMock()
    mock_main_branch.name = "main"
    mock_main_branch.target_commit = "fakecommithash"
    mock_branch = MagicMock()
    mock_branch.name = "dev"
    mock_branch.target_commit = "wrongfakehash"

    # Mock the response from HuggingFace
    mock_response = MagicMock()
    mock_response.branches = [mock_main_branch, mock_branch, mock_branch]
    mocker.patch(
        "garden_ai.model_connectors.hugging_face.hfh.list_repo_refs",
        return_value=mock_response,
    )

    repo = "fake/repo"
    connector = HFConnector(repo)

    # The revision field should be the commit has from the latest commit on main
    assert connector.revision == "fakecommithash"

    # The commit hash should be stored in the metadata object as well
    assert connector.metadata.model_version == "fakecommithash"

    # The commit hash should be passed into hfh.snapshot_download when stage is called
    connector.stage()
    mock_snapshot_download.assert_called_with(
        "fake/repo", revision="fakecommithash", local_dir="hf_model"
    )
