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
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    args = mock_client.register_model.call_args.args
    local_model = args[0]
    assert local_model.local_path == str(tmp_path)
    assert local_model.model_name == "unit-test-model"
    assert local_model.flavor == "sklearn"


@pytest.mark.cli
def test_model_add_dataset(mocker, second_draft_of_model):
    registered_model = second_draft_of_model
    mocker.patch(
        "garden_ai.local_data.get_local_model_by_name"
    ).return_value = registered_model
    command = [
        "model",
        "add-dataset",
        "--model",
        str(second_draft_of_model.full_name),
        "--title",
        "fake dataset",
        "--url",
        "example.com/123456",
        "--doi",
        "uc-435t/abcde",
        "--datatype",
        ".csv",
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0
    assert registered_model.dataset.title == "fake dataset"
    assert registered_model.dataset.url == "example.com/123456"
    assert registered_model.dataset.data_type == ".csv"
    assert registered_model.dataset.doi == "uc-435t/abcde"


@pytest.mark.cli
def test_model_list(database_with_connected_pipeline, tmp_path, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_pipeline
    )

    model_full_name = "willengler@uchicago.edu/will-test-model"
    model_name = "will-test-model"
    model_flavor = "sklearn"

    command = [
        "model",
        "list",
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    assert model_full_name in result.stdout
    assert model_name in result.stdout
    assert model_flavor in result.stdout


@pytest.mark.cli
def test_model_show(database_with_connected_pipeline, tmp_path, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_pipeline
    )

    model_full_name = "willengler@uchicago.edu/will-test-model"
    model_name = "will-test-model"
    model_flavor = "sklearn"

    command = [
        "model",
        "show",
        model_full_name,
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    assert model_full_name in result.stdout
    assert model_name in result.stdout
    assert model_flavor in result.stdout

    command = [
        "model",
        "show",
        "not_a_model_name",
        model_full_name,
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0

    assert model_full_name in result.stdout
    assert model_name in result.stdout
    assert model_flavor in result.stdout
