import base64
import json
import unittest
from unittest.mock import MagicMock

from garden_ai import (
    EntrypointMetadata,
    PublishedGarden,
    garden_entrypoint,
    garden_step,
)
from garden_ai.model_connectors import GitHubConnector, HFConnector
from garden_ai.model_connectors.exceptions import ConnectorLFSError
from garden_ai.model_connectors.model_utils import create_connector


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


def test_garden_entrypoint_decorator():
    entrypoint_meta = EntrypointMetadata(
        title="My Entrypoint",
        authors=["Willie", "Waylon", "Johnny", "Kris"],
        description="A garden entrypoint",
        tags=["garden_ai"],
    )

    model_connector = HFConnector(repo_id="willengler-uc/iris-classifier")

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

    model_connector = GitHubConnector(repo_id="uw-cmg/ASR_model")

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
    assert models[0].model_identifier == "uw-cmg/ASR_model"
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
    mock_repo_base = mocker.patch("garden_ai.model_connectors.model_connector.Repo")
    mock_repo_instance = MagicMock()
    mock_repo_class.return_value = mock_repo_instance
    mock_repo_base.return_value = mock_repo_instance
    mock_repo_instance.remotes.origin.pull = MagicMock()
    # Set up the mock to return a URL that matches the connector's repo_url
    mock_repo_instance.remotes.origin.url = "https://github.com/fake/repo.git"

    # Mock Repo.clone_from method to track calls without actually cloning
    mock_clone_from = mocker.patch(
        "garden_ai.model_connectors.github_conn.Repo.clone_from"
    )

    # Mock is_git_dir to control the flow in the stage method
    mocker.patch(
        "garden_ai.model_connectors.model_connector.is_git_dir",
        side_effect=[
            False,
            True,
            True,
        ],  # First call: not a git dir, then it is a git dir
    )

    fake_commit_hash = "a" * 40

    # enable_imports=False just bc mocking sys.path.append was hard
    connector = GitHubConnector(
        repo_url="https://github.com/fake/repo",
        enable_imports=False,
        revision=fake_commit_hash,
    )

    # First call should trigger clone since it's not a git dir yet
    connector.stage()
    mock_clone_from.assert_called_once_with(
        "https://github.com/fake/repo.git", "models/repo", branch="main"
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
    fake_commit_hash = "a" * 40
    mock_response = mocker.patch("requests.get")
    mock_response.return_value.json.return_value = {
        "sha": fake_commit_hash,
        "content": "",
    }

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
    assert connector.revision == fake_commit_hash

    # The commit hash should be stored in the metadata object as well
    assert connector.metadata.model_version == fake_commit_hash

    # stage should checkout the pinned commit when called
    connector.stage()
    mock_repo.git.checkout.assert_called_once_with(fake_commit_hash)


def test_HFConnector_pins_commit_hash_correctly(mocker):
    mocker.patch("os.path.exists", return_value=True)
    mock_snapshot_download = mocker.patch(
        "garden_ai.model_connectors.hugging_face.hfh.snapshot_download"
    )

    fake_commit_hash = "a" * 40

    # Mock a couple of hfh.GitRefInfo objects, these are returned from hfh.list_repo_refs
    mock_main_branch = MagicMock()
    mock_main_branch.name = "main"
    mock_main_branch.target_commit = fake_commit_hash
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
    connector = HFConnector(repo_id=repo)

    # The revision field should be the commit has from the latest commit on main
    assert connector.revision == fake_commit_hash

    # The commit hash should be stored in the metadata object as well
    assert connector.metadata.model_version == fake_commit_hash

    # The commit hash should be passed into hfh.snapshot_download when stage is called
    connector.stage()
    mock_snapshot_download.assert_called_with(
        repo_id="fake/repo",
        local_dir="models/repo",
        revision=fake_commit_hash,
    )


def test_GitHubConnector_raises_exception_with_lfs_file(mocker):
    # Mock the response from the GitHub API
    mock_response = MagicMock()
    mock_response.status_code = 200
    # The file contents are base64 encoded
    git_lfs_attributes = "large_file.txt filter=lfs diff=lfs merge=lfs -text"
    git_lfs_attributes_bytes = git_lfs_attributes.encode("utf-8")
    mock_response.json.return_value = {
        "content": base64.b64encode(git_lfs_attributes_bytes)
    }
    mocker.patch(
        "garden_ai.model_connectors.github_conn.requests.get",
        return_value=mock_response,
    )

    # Constructing a connector using a repo with a git-lfs file should raise an error
    with unittest.TestCase().assertRaises(ConnectorLFSError):
        GitHubConnector(
            repo_url="https://github.com/fake/repo",
            revision="a" * 40,
            readme="",  # give it a readme so it doesn't try to pull one from the repo
        )


def test_create_connector_returns_correct_connector_type():
    fake_revision = "a" * 40

    # Give them a fake revision so they don't attempt to fetch one.
    gh_url = create_connector("https://github.com/fake/repo", revision=fake_revision)
    assert isinstance(gh_url, GitHubConnector)

    hf_url = create_connector(
        "https://huggingface.co/fake/repo",
        revision=fake_revision,
    )
    assert isinstance(hf_url, HFConnector)


def test_create_connector_with_url():
    fake_revision = "a" * 40

    hf = create_connector("https://huggingface.co/real/repo", revision=fake_revision)
    assert isinstance(hf, HFConnector)
    assert hf.repo_id == "real/repo"

    gh = create_connector("https://github.com/real/repo", revision=fake_revision)
    assert isinstance(gh, GitHubConnector)
    assert gh.repo_id == "real/repo"

    # URL must be full HTTP e.g. "https://github.com/owner/repo"
    with unittest.TestCase().assertRaises(Exception):
        create_connector("github.com/bad/repo")

    with unittest.TestCase().assertRaises(Exception):
        create_connector("huggingface.co/bad/repo")
