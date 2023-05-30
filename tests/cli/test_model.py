import pytest
from typer.testing import CliRunner

from garden_ai.app.main import app
from garden_ai.client import GardenClient

from typer import BadParameter

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
def test_scaffolded_model_upload(mocker, tmp_path):
    user_email = "test@example.com"
    mock_client = mocker.MagicMock(GardenClient)
    mocker.patch("garden_ai.app.model.GardenClient").return_value = mock_client
    mock_client.get_email.return_value = user_email
    command = [
        "model",
        "register",
        "YOUR MODEL's NAME HERE",
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

    # check that a user can't register a model with name "YOUR MODEL's NAME HERE"
    result = runner.invoke(app, command)
    assert result.exit_code == 2
