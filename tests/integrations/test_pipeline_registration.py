import pytest
from typer.testing import CliRunner

from garden_ai.app.main import app
from garden_ai.utils.filesystem import load_pipeline_from_python_file
from tests.fixtures.helpers import get_fixture_file_path  # type: ignore

runner = CliRunner()


@pytest.mark.integration
def test_register_pipeline():
    fixture_pipeline_path = get_fixture_file_path(
        "fixture_pipeline/integration_pipeline.py"
    )
    command = "pipeline register " + str(fixture_pipeline_path)
    result = runner.invoke(app, command)
    assert result.exit_code == 0


@pytest.mark.integration
def test_publish_garden():
    # fixture_pipeline_path = get_fixture_file_path("fixture_pipeline/pipeline.py")
    command = "garden publish -g e1a3b50b-4efc-42c8-8422-644f4f858b87"
    result = runner.invoke(app, command)
    assert result.exit_code == 0


@pytest.mark.integration
def test_run_registered_pipeline():
    fixture_pipeline_path = get_fixture_file_path(
        "fixture_pipeline/integration_pipeline.py"
    )
    identity_pipeline = load_pipeline_from_python_file(fixture_pipeline_path)
    dlhub_endpoint = "86a47061-f3d9-44f0-90dc-56ddc642c000"  # real endpoint
    input_data = "I'm afraid of change"
    output_data = identity_pipeline(input_data, endpoint=dlhub_endpoint)
    assert input_data == output_data
