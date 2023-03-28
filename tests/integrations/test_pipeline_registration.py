import pytest

from typer.testing import CliRunner

from garden_ai.app.main import app
from tests.fixtures.helpers import get_fixture_file_path  # type: ignore


runner = CliRunner()


@pytest.mark.integration
def test_register_pipeline():
    fixture_pipeline_path = get_fixture_file_path("fixture_pipeline/pipeline.py")
    command = "pipeline register " + str(fixture_pipeline_path)
    result = runner.invoke(app, command)
    assert result.exit_code == 0
