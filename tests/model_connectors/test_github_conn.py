import base64
from pathlib import Path
import pytest

from garden_ai.model_connectors import GitHubConnector, create_connector
from garden_ai.model_connectors.exceptions import (
    ConnectorLFSError,
    ConnectorInvalidRepoIdError,
)


def test_GHconnector_stage_is_idempotent(
    mocker,
    patch_has_lfs,
    patch_fetch_readme,
):
    # Mock os.path.exists and os.mkdir
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("os.mkdir")

    # Mock Repo to simulate both clone and pull scenarios
    mock_repo_class = mocker.patch("garden_ai.model_connectors.github_conn.Repo")
    mock_repo_base = mocker.patch("garden_ai.model_connectors.model_connector.Repo")
    mock_repo_instance = mocker.MagicMock()
    mock_repo_class.return_value = mock_repo_instance
    mock_repo_base.return_value = mock_repo_instance
    mock_repo_instance.remotes.origin.pull = mocker.MagicMock()
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


def test_GitHubConnector_pins_commit_hash_correctly(
    mocker,
    patch_has_lfs,
    patch_fetch_readme,
):
    # Mock the response from the GitHub API
    fake_commit_hash = "a" * 40
    mock_response = mocker.patch("requests.get")
    mock_response.return_value.json.return_value = {
        "sha": fake_commit_hash,
        "content": "",
    }

    # Mock the Repo
    repo_url = "https://github.com/fake/repo"
    mock_repo = mocker.MagicMock()
    mock_repo.remotes.origin.url = repo_url + ".git"
    mock_repo.remotes.origin.pull = mocker.MagicMock()
    mock_repo.git.checkout = mocker.MagicMock()
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


def test_GitHubConnector_raises_exception_with_lfs_file(mocker):
    # Mock the response from the GitHub API
    mock_response = mocker.MagicMock()
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
    with pytest.raises(ConnectorLFSError):
        GitHubConnector(
            repo_url="https://github.com/fake/repo",
            revision="a" * 40,
            readme="",  # give it a readme so it doesn't try to pull one from the repo
        )


def test_init_raises_on_non_github_url(
    patch_fetch_readme,
    patch_infer_revision,
):
    with pytest.raises(ConnectorInvalidRepoIdError):
        GitHubConnector(
            repo_url="https://some.repo.example.com",
        )


@pytest.mark.integration
def test_stage_works_on_empty_local_dir(
    tmp_path,
):
    """This tests calling `stage` on a GitHubConnector when the local directory is empty.

    This test may fail if there is no internet access or github.com is unreachable.

    This was created as part of the resolution of this issue:
    https://github.com/Garden-AI/garden/issues/521
    """
    c = create_connector(
        "https://github.com/WillEngler/model-perovskite-ASR", local_dir=tmp_path
    )
    assert isinstance(c, GitHubConnector)

    # This might still throw errors if the network is down or github is unreachable
    model_dir = c.stage()

    # Make sure we have cloned the repo
    git_dir = Path(model_dir) / ".git"
    assert git_dir.exists()

    # Ensure we have checked out the correct revision
    head = git_dir / "HEAD"
    with open(head, "r") as f:
        head_rev = f.readline().strip()
    assert head_rev == c.revision
