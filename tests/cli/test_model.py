import pytest
from typer.testing import CliRunner

from garden_ai.app.main import app
from garden_ai.client import GardenClient

runner = CliRunner()


@pytest.mark.cli
def test_model_upload(mocker, tmp_path):
    user_email = "test@example.com"
    mock_client = mocker.MagicMock(GardenClient)
    mocker.patch("garden_ai.app.model.GardenClient").return_value = mock_client
    mock_client.get_email.return_value = user_email
    command = [
        "model",
        "register",
        "unit-test-model",
        str(tmp_path),
        "sklearn",
        "--extra-pip-requirements",
        "torch==1.13.1",
        "--extra-pip-requirements",
        "pandas<=1.5.0",
        "--dataset-url",
        "example.com/123456",
        "--dataset-doi",
        "uc-435t/abcde",
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    args = mock_client.register_model.call_args.args
    local_model = args[0]
    assert local_model.local_path == str(tmp_path)
    assert local_model.model_name == "unit-test-model"
    assert local_model.flavor == "sklearn"
    assert local_model.extra_pip_requirements == ["torch==1.13.1", "pandas<=1.5.0"]
    dataset_connection = local_model.connections[0]
    assert dataset_connection.doi == "uc-435t/abcde"
    assert dataset_connection.url == "example.com/123456"
    assert dataset_connection.repository == "Foundry"


@pytest.mark.cli
def test_model_list(database_with_connected_pipeline, tmp_path, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_pipeline
    )

    model_uri = "willengler@uchicago.edu-will-test-model/3"
    model_name = "will-test-model"
    model_flavor = "sklearn"

    command = [
        "model",
        "list",
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    assert model_uri in result.stdout
    assert model_name in result.stdout
    assert model_flavor in result.stdout


@pytest.mark.cli
def test_model_show(database_with_connected_pipeline, tmp_path, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_pipeline
    )

    model_uri = "willengler@uchicago.edu-will-test-model/3"
    model_name = "will-test-model"
    model_flavor = "sklearn"

    command = [
        "model",
        "show",
        model_uri,
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    assert model_uri in result.stdout
    assert model_name in result.stdout
    assert model_flavor in result.stdout

    command = [
        "model",
        "show",
        model_uri,
        model_uri,
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    assert model_uri in result.stdout
    assert model_name in result.stdout
    assert model_flavor in result.stdout
