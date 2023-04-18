import json

from garden_ai import local_data


def test_local_storage_garden(mocker, garden_client, garden_all_fields, tmp_path):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.client.LOCAL_STORAGE", new=tmp_path)
    uuid = garden_all_fields.uuid
    local_data.put_local_garden(garden_all_fields)
    record = local_data.get_local_garden_by_uuid(uuid)
    assert json.dumps(record) == garden_all_fields.json()


def test_local_storage_pipeline(mocker, garden_client, pipeline_toy_example, tmp_path):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.client.LOCAL_STORAGE", new=tmp_path)
    uuid = pipeline_toy_example.uuid
    local_data.put_local_pipeline(pipeline_toy_example)
    record = local_data.get_local_pipeline_by_uuid(uuid)
    assert json.dumps(record) == pipeline_toy_example.json()


def test_local_storage_keyerror(mocker, garden_client, garden_all_fields, tmp_path):
    # mock to replace "~/.garden/db"
    tmp_path.mkdir(parents=True, exist_ok=True)
    mocker.patch("garden_ai.client.LOCAL_STORAGE", new=tmp_path)
    pipeline, *_ = garden_all_fields.pipelines
    # put the pipeline, not garden (hence db is nonempty)
    local_data.put_local_pipeline(pipeline)

    # can't find the garden
    assert local_data.get_local_garden_by_uuid(garden_all_fields.uuid) is None

    # can find the pipeline
    record = local_data.get_local_pipeline_by_uuid(pipeline.uuid)
    assert json.dumps(record) == pipeline.json()
