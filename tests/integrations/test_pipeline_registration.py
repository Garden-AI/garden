import pytest
from typer.testing import CliRunner

from garden_ai import GardenClient
from garden_ai.app.main import app
from garden_ai.utils.filesystem import load_pipeline_from_python_file
from tests.fixtures.helpers import get_fixture_file_path  # type: ignore

runner = CliRunner()


# @pytest.fixture(autouse=False)
# def auto_mock_GardenClient_set_up_mlflow_env():
#     # actually, DO set mlflow env variables
#     # locally defining a fixture to override
#     # the one mocking `setup_mlflow_env`
#     pass


@pytest.fixture
def dlhub_endpoint():
    return "86a47061-f3d9-44f0-90dc-56ddc642c000"  # real endpoint


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
def test_register_and_run_with_env_vars(dlhub_endpoint):
    client = GardenClient()
    # register the pipeline
    pipeline_path = get_fixture_file_path("fixture_pipeline/env_vars_pipeline.py")
    # NOTE: this pipeline just returns a copy of the os.environ dictionary
    pipeline = load_pipeline_from_python_file(pipeline_path)
    container_id = "3dc3170e-2cdc-4379-885d-435a0d85b581"
    client.register_pipeline(pipeline, container_id)

    # load from client, setting mlflow env vars
    registered_pipeline = client.get_registered_pipeline(pipeline.doi)

    input_data = "i'm afraid of change"
    # set arbitrary other env vars
    test_var, test_val = (
        "GARDEN_TEST_ENV_VARIABLE",
        "... especially of a changing environment",
    )
    registered_pipeline._env_vars[test_var] = test_val

    results = registered_pipeline(input_data, endpoint=dlhub_endpoint)
    assert results[test_var] == test_val
    assert "MLFLOW_TRACKING_URI" in results
    assert "MLFLOW_TRACKING_TOKEN" in results


@pytest.mark.integration
def test_register_and_run_with_model(dlhub_endpoint):
    client = GardenClient()
    # register the pipeline locally
    pipeline_path = get_fixture_file_path("fixture_pipeline/soup_pipeline.py")
    # NOTE: this pipeline references a model, and raises an exception if model
    # fails to lazy-download when model.predict() is called
    pipeline = load_pipeline_from_python_file(pipeline_path)
    container_id = "3dc3170e-2cdc-4379-885d-435a0d85b581"
    client.register_pipeline(pipeline, container_id)

    # load from client, setting mlflow env vars
    registered_pipeline = client.get_registered_pipeline(pipeline.doi)

    input_soup = [1, 2, 3]
    assert registered_pipeline(input_soup, endpoint=dlhub_endpoint) == (10 / 10)
