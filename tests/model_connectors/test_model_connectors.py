import pytest

from garden_ai.model_connectors import GitHubConnector, HFConnector
from garden_ai.model_connectors.model_utils import create_connector


def test_create_connector_returns_correct_connector_type(
    patch_has_lfs,
    patch_fetch_readme,
):
    fake_revision = "a" * 40

    # Give them a fake revision so they don't attempt to fetch one.
    gh_url = create_connector("https://github.com/fake/repo", revision=fake_revision)
    assert isinstance(gh_url, GitHubConnector)

    hf_url = create_connector(
        "https://huggingface.co/fake/repo",
        revision=fake_revision,
    )
    assert isinstance(hf_url, HFConnector)


def test_create_connector_raises_on_invalid_url():
    # URL must be full HTTP e.g. "https://github.com/owner/repo"
    with pytest.raises(Exception):
        create_connector("github.com/bad/repo")

    with pytest.raises(Exception):
        create_connector("huggingface.co/bad/repo")


def test_create_connector_parses_repo_id(
    patch_has_lfs,
    patch_fetch_readme,
):
    fake_revision = "a" * 40

    hf = create_connector("https://huggingface.co/real/repo", revision=fake_revision)
    assert isinstance(hf, HFConnector)
    assert hf.repo_id == "real/repo"

    gh = create_connector("https://github.com/real/repo", revision=fake_revision)
    assert isinstance(gh, GitHubConnector)
    assert gh.repo_id == "real/repo"
