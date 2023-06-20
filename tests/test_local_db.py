from garden_ai import local_data


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
    local_data.put_local_model(second_draft_of_model)

    # New version of same model name should get overwritten in local DB
    registered_model = local_data.get_local_model_by_uri(
        second_draft_of_model.model_uri
    )
    assert registered_model is not None
    overwritten_model = local_data.get_local_model_by_uri(
        "test@example.com-unit-test-model/1"
    )
    assert overwritten_model is None
