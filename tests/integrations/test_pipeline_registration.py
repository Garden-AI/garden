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


# @pytest.mark.integration
# def test_register_pipeline():
#     # fixture_pipeline_path = get_fixture_file_path("fixture_pipeline/pipeline.py")
#     command = "garden publish -g e1a3b50b-4efc-42c8-8422-644f4f858b87"
#     result = runner.invoke(app, command)
#     assert result.exit_code == 0
