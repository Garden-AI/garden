import pytest
from typer.testing import CliRunner
from typing import Optional

from garden_ai import GardenClient, RegisteredPipeline
from garden_ai.app.main import app
from garden_ai.local_data import get_local_pipeline_by_uuid
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


@pytest.mark.integration
def test_run_registered_pipeline_with_env_vars():
    pipeline_path = get_fixture_file_path("fixture_pipeline/env_vars_pipeline.py")
    pipeline = load_pipeline_from_python_file(pipeline_path)
    # note: env vars pipeline just returns `dict(os.environ)`
    registered_pipeline = get_local_pipeline_by_uuid(pipeline.uuid)
    if not registered_pipeline:
        # note: currently we only run integration tests locally, so check before
        # re-registering the test pipeline in case its already been registered_pipeline
        container_id = "3dc3170e-2cdc-4379-885d-435a0d85b581"
        command = "pipeline register " + str(pipeline_path)
        runner.invoke(app, command)
        client = GardenClient()
        client.register_pipeline(pipeline, container_id)
        registered_pipeline = get_local_pipeline_by_uuid(pipeline.uuid)
        assert registered_pipeline is not None

    dlhub_endpoint = "86a47061-f3d9-44f0-90dc-56ddc642c000"  # real endpoint
    input_data = "i'm afraid of change"

    test_var, test_val = (
        "GARDEN_TEST_ENV_VARIABLE",
        "... especially of a changing environment",
    )
    registered_pipeline._env_vars = {test_var: test_val}
    results = registered_pipeline(input_data, endpoint=dlhub_endpoint)
    assert results[test_var] == test_val
