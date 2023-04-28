import pytest
from typer.testing import CliRunner

from garden_ai.app.main import app
from garden_ai.client import GardenClient

runner = CliRunner()


@pytest.mark.cli
def test_model_upload(mocker, tmp_path):
    mock_client = mocker.MagicMock(GardenClient)
    mocker.patch("garden_ai.app.model.GardenClient").return_value = mock_client
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
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    args = mock_client.log_model.call_args.args
    assert args[0] == str(tmp_path)
    assert args[1] == "unit-test-model"
    assert args[2] == "sklearn"
    assert args[3] == ["torch==1.13.1", "pandas<=1.5.0"]
