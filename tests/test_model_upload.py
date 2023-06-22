import os

from tests.fixtures.helpers import get_fixture_file_path  # type: ignore
from garden_ai.mlmodel import stage_model_for_upload, LocalModel


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
