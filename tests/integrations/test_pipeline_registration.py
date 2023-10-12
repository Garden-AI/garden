import pytest
from typer.testing import CliRunner

from garden_ai.app.main import app

runner = CliRunner()


@pytest.fixture
def dlhub_endpoint():
    return "86a47061-f3d9-44f0-90dc-56ddc642c000"  # real endpoint


@pytest.mark.integration
def test_publish_garden():
    command = "garden publish -g e1a3b50b-4efc-42c8-8422-644f4f858b87"
    result = runner.invoke(app, command)
    assert result.exit_code == 0
