import pathlib  # noqa
from unittest.mock import patch

import pytest

from garden_ai._version import __version__
from garden_ai.constants import GardenConstants


@pytest.mark.cli
def test_version_displays_correctly(cli_runner, app):
    result = cli_runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"garden-ai {__version__}" in result.stdout


@pytest.mark.cli
def test_whoami_prints_logged_in_user(cli_runner, app):
    result = cli_runner.invoke(app, ["whoami"])
    assert result.exit_code == 0
    assert "fake@email.com" in result.stdout


@pytest.mark.cli
def test_login_prints_message_if_logged_in(cli_runner, app, logged_in_user):
    result = cli_runner.invoke(app, ["login"])
    assert result.exit_code == 0
    assert "Already logged in" in result.stdout


@pytest.mark.cli
def test_logout_deletes_key_store(cli_runner, app, tmp_path):
    with patch.object(
        GardenConstants, "GARDEN_KEY_STORE", tmp_path / "tokens.json"
    ) as key_store:
        # Make sure the key store exists
        key_store.touch()
        assert key_store.exists()

        # Logout and make sure the key store was deleted
        result = cli_runner.invoke(app, ["logout"])
        assert result.exit_code == 0
        assert not key_store.exists()


@pytest.mark.cli
def test_print_usage_info_if_no_args(cli_runner, app):
    result = cli_runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Hello, Garden" in result.stdout
    assert "Usage:" in result.stdout
