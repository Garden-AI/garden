from garden_ai import local_data
from garden_ai.utils.misc import get_cache_tag


def test_local_storage_garden(mocker, garden_client, garden_all_fields, tmp_path):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=tmp_path)
    doi = garden_all_fields.doi
    local_data.put_local_garden(garden_all_fields)
    from_record = local_data.get_local_garden_by_doi(doi)
    assert from_record == garden_all_fields


def test_local_storage_pipeline(
    mocker, garden_client, registered_pipeline_toy_example, tmp_path
):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=tmp_path)
    local_data.put_local_pipeline(registered_pipeline_toy_example)
    doi = registered_pipeline_toy_example.doi
    from_record = local_data.get_local_pipeline_by_doi(doi)
    assert from_record == registered_pipeline_toy_example


def test_local_storage_keyerror(
    mocker, garden_client, registered_pipeline_toy_example, garden_all_fields, tmp_path
):
    # mock to replace "~/.garden/db"
    tmp_path.mkdir(parents=True, exist_ok=True)
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=tmp_path)
    # put the pipeline, not the garden
    pipeline = registered_pipeline_toy_example
    local_data.put_local_pipeline(pipeline)

    # can't find the garden
    assert local_data.get_local_garden_by_doi(garden_all_fields.doi) is None

    # can find the pipeline
    from_record = local_data.get_local_pipeline_by_doi(pipeline.doi)
    assert from_record == pipeline


def test_local_storage_model(mocker, database_with_model, second_draft_of_model):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=database_with_model)

    # Starts out as sklearn model
    orginal_model = local_data.get_local_model_by_name(second_draft_of_model.full_name)
    assert orginal_model.flavor == "sklearn"

    # New version of same model name should get overwritten in local DB
    local_data.put_local_model(second_draft_of_model)
    overwritten_model = local_data.get_local_model_by_name(
        second_draft_of_model.full_name
    )
    assert overwritten_model.flavor == "pytorch"


def test_local_cache(mocker, garden_client, pipeline_toy_example):
    reqs_a = ["tensorflow", "pandas", "mlflow==2.5.0"]
    reqs_dup_a = ["tensorflow", "pandas", "mlflow==2.5.0", "pandas"]
    reqs_reorder_a = ["mlflow==2.5.0", "tensorflow", "pandas"]
    reqs_b = ["pandas<3", "opencv-python", "mlflow==2.4.2"]

    py_version = "3.10.8"

    assert (
        get_cache_tag(reqs_a, [], py_version)
        == get_cache_tag(reqs_dup_a, [], py_version)
        == get_cache_tag(reqs_reorder_a, [], py_version)
    )
    assert get_cache_tag(reqs_a, [], py_version) != get_cache_tag(
        reqs_b, [], py_version
    )
    assert get_cache_tag(reqs_a, [], py_version) != get_cache_tag(
        reqs_a, ["not-equal"], py_version
    )
    assert get_cache_tag(reqs_a, [], py_version) != get_cache_tag(reqs_a, [], "3.10.9")

    build_method = mocker.patch(
        "garden_ai.client.build_container",
        return_value="d1fc6d30-be1c-4ac4-a289-d87b27e84357",
    )
    mocker.patch(
        "garden_ai.client._read_local_cache",
        return_value={
            get_cache_tag(
                reqs_a, [], py_version
            ): "d1fc6d30-be1c-4ac4-a289-d87b27e84357"
        },
    )
    mocker.patch("garden_ai.client._write_local_cache", return_value=None)
    mocker.patch(
        "garden_ai.client.register_pipeline",
        return_value="9f5688ac-424d-443e-b525-97c72e4e083f",
    )
    mocker.patch("garden_ai.client.GardenClient._update_datacite", return_value=None)
    mocker.patch("garden_ai.client.local_data.put_local_pipeline", return_value=None)

    pipeline_toy_example.pip_dependencies = reqs_a
    garden_client.register_pipeline(pipeline_toy_example)

    assert build_method.call_count == 0

    pipeline_toy_example.pip_dependencies = ["opencv-python"]
    garden_client.register_pipeline(pipeline_toy_example)

    assert build_method.call_count == 1
