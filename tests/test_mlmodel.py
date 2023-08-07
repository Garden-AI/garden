import json
import os
import pytest

from tests.fixtures.helpers import get_fixture_file_path  # type: ignore
from garden_ai.mlmodel import (
    stage_model_for_upload,
    LocalModel,
    _Model,
    ModelUploadException,
)
from garden_ai.backend_client import BackendClient, PresignedUrlResponse
from garden_ai.model_file_transfer.upload import upload_mlmodel_to_s3


def test_stage_model_for_upload(mocker, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    mocker.patch("garden_ai.mlmodel.MODEL_STAGING_DIR", new=tmp_path)
    model_path = get_fixture_file_path("fixture_models/iris_model.pkl")
    print(model_path)
    local_model = LocalModel(
        model_name="test_model",
        flavor="sklearn",
        local_path=str(model_path),
        user_email="willengler@uchicago.edu",
    )
    staged_path = stage_model_for_upload(local_model)
    assert staged_path.endswith("/artifacts/model")
    files_to_check = ["MLmodel", "model.pkl", "requirements.txt"]

    for filename in files_to_check:
        assert os.path.isfile(
            os.path.join(staged_path, filename)
        ), f"File {filename} does not exist."


def test_upload_model_to_s3(mocker, local_model, tmp_path):
    mock_upload = mocker.patch(
        "garden_ai.model_file_transfer.upload._upload_directory_to_s3_presigned"
    )
    fake_staging_dir = str(tmp_path)

    backend_client = mocker.MagicMock(BackendClient)
    mock_fields = {"key": "ABC"}
    mock_presigned_url_response = PresignedUrlResponse(
        "test@example.com/unit-test-model", "foo.com", mock_fields
    )
    backend_client.get_model_upload_url.return_value = mock_presigned_url_response

    upload_mlmodel_to_s3(fake_staging_dir, local_model, backend_client)
    backend_client.get_model_upload_url.assert_called_with(
        "test@example.com/unit-test-model"
    )
    mock_upload.assert_called_with(fake_staging_dir, mock_presigned_url_response)


def test_get_download_url(model_url_env_var):
    url = _Model.get_download_url("willengler@uchicago.edu/test_model")
    assert url == "presigned-url.aws.com"


def test_invalid_extra_paths(mocker, local_model, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    mocker.patch("garden_ai.mlmodel.MODEL_STAGING_DIR", new=tmp_path)
    model_path = get_fixture_file_path("fixture_models/iris_model.pkl")
    local_model = LocalModel(
        model_name="test_model",
        flavor="sklearn",
        local_path=str(model_path),
        user_email="willengler@uchicago.edu",
        extra_paths=["invalid-path.py"],
    )
    with pytest.raises(ModelUploadException):
        stage_model_for_upload(local_model)


def test_extra_paths(mocker, local_model, tmp_path):
    mocker.patch("garden_ai.mlmodel.MODEL_STAGING_DIR", new=tmp_path)
    model_path = get_fixture_file_path("fixture_models/pytorchtest.pth")
    file_path = get_fixture_file_path("fixture_models/torch.py")
    local_model = LocalModel(
        model_name="test_model",
        flavor="pytorch",
        local_path=str(model_path),
        user_email="willengler@uchicago.edu",
        extra_paths=[str(file_path)],
    )
    staged_path = stage_model_for_upload(local_model)
    assert staged_path.endswith("/artifacts/model")


def test_load_model_before_predict(mocker, model_url_env_var, mlflow_metadata):
    from mlflow.pyfunc import PyFuncModel  # type: ignore

    mock_pyfunc_model = mocker.MagicMock(PyFuncModel)
    mocker.patch("mlflow.pyfunc.load_model").return_value = mock_pyfunc_model
    mocker.patch(
        "garden_ai._model._Model.download_and_stage"
    ).return_value = (
        "/Users/FakeUser/.garden/mlflow/willengler@uchicago.edu/test_model/unzipped"
    )
    mocker.patch("builtins.open", mocker.mock_open(read_data=mlflow_metadata))
    model = _Model("willengler@uchicago.edu/test_model")
    assert model.model is not None
    assert isinstance(model.model, PyFuncModel)


def test_load_model_before_predict_torch(
    mocker, model_url_env_var, mlflow_metadata_torch
):
    from mlflow.pyfunc import PyFuncModel  # type: ignore

    # Mock mlflow.pytorch.load_model with PyFuncModel output instead of torch.nn.Module
    # Done to avoid importing pytorch to test suite
    mock_torch_model = mocker.MagicMock(PyFuncModel)
    mocker.patch("mlflow.pytorch.load_model").return_value = mock_torch_model
    mocker.patch(
        "garden_ai._model._Model.download_and_stage"
    ).return_value = (
        "/Users/FakeUser/.garden/mlflow/willengler@uchicago.edu/test_model/unzipped"
    )
    mocker.patch("builtins.open", mocker.mock_open(read_data=mlflow_metadata_torch))
    model = _Model("willengler@uchicago.edu/test_model")
    assert model.model is not None
    assert isinstance(model.model, _Model._TorchWrapper)
    assert isinstance(model.model._wrapped_model, PyFuncModel)


def test_generate_presigned_urls_for_garden(
    mocker, garden_client, garden_all_fields, registered_pipeline_toy_example
):
    registered_pipeline_toy_example.model_full_names = [
        "willengler@uchicago.edu/test_model"
    ]
    mock_presigned_url_response = PresignedUrlResponse(
        "willengler@uchicago.edu/test_model", "presigned-url.aws.com", {}
    )
    mocker.patch.object(
        garden_client.backend_client,
        "get_model_download_urls",
        return_value=[mock_presigned_url_response],
    )
    garden_all_fields._pipelines = [registered_pipeline_toy_example]

    garden_client._generate_presigned_urls_for_garden(garden_all_fields)
    env_var_string = garden_all_fields.pipelines[0]._env_vars["GARDEN_MODELS"]
    as_dict = json.loads(env_var_string)
    assert as_dict["willengler@uchicago.edu/test_model"] == "presigned-url.aws.com"
