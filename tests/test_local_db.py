from garden_ai import local_data


def test_local_storage_garden(mocker, garden_client, garden_all_fields, tmp_path):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=tmp_path)
    uuid = garden_all_fields.uuid
    local_data.put_local_garden(garden_all_fields)
    from_record = local_data.get_local_garden_by_uuid(uuid)
    assert from_record == garden_all_fields


def test_local_storage_pipeline(
    mocker, garden_client, registered_pipeline_toy_example, tmp_path
):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=tmp_path)
    local_data.put_local_pipeline(registered_pipeline_toy_example)
    uuid = registered_pipeline_toy_example.uuid
    from_record = local_data.get_local_pipeline_by_uuid(uuid)
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
    assert local_data.get_local_garden_by_uuid(garden_all_fields.uuid) is None

    # can find the pipeline
    from_record = local_data.get_local_pipeline_by_uuid(pipeline.uuid)
    assert from_record == pipeline
