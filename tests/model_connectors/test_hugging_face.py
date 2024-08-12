import pytest

from garden_ai.model_connectors import HFConnector
from garden_ai.model_connectors.exceptions import ConnectorInvalidRepoIdError


def test_HFConnector_pins_commit_hash_correctly(
    mocker,
    patch_fetch_readme,
):
    mocker.patch("os.path.exists", return_value=True)
    mock_snapshot_download = mocker.patch(
        "garden_ai.model_connectors.hugging_face.hfh.snapshot_download"
    )

    fake_commit_hash = "a" * 40

    # Mock a couple of hfh.GitRefInfo objects, these are returned from hfh.list_repo_refs
    mock_main_branch = mocker.MagicMock()
    mock_main_branch.name = "main"
    mock_main_branch.target_commit = fake_commit_hash
    mock_branch = mocker.MagicMock()
    mock_branch.name = "dev"
    mock_branch.target_commit = "wrongfakehash"

    # Mock the response from HuggingFace
    mock_response = mocker.MagicMock()
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


def test_init_raises_on_non_huggingface_url(
    patch_fetch_readme,
    patch_infer_revision,
):
    with pytest.raises(ConnectorInvalidRepoIdError):
        HFConnector(repo_url="https://not.a.hf-repo.example.com")
