from garden_ai import local_data
from rich.table import Table


def test_local_storage_garden(mocker, garden_client, garden_all_fields, tmp_path):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=tmp_path)
    uuid = garden_all_fields.uuid
    local_data.put_local_garden(garden_all_fields)
    from_record = local_data.get_local_garden_by_uuid(uuid)
    assert from_record == garden_all_fields


def test_get_local_garden_json(mocker, garden_client, garden_all_fields, tmp_path):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=tmp_path)
    local_data.put_local_garden(garden_all_fields)

    garden_data = local_data._read_local_db().get(
        local_data.ResourceType.GARDEN.value, {}
    )[str(garden_all_fields.uuid)]

    assert garden_data == local_data.get_local_garden_json(str(garden_all_fields.uuid))
    assert garden_data == local_data.get_local_garden_json(str(garden_all_fields.doi))


def test_get_local_garden_table(mocker, garden_client, garden_all_fields, tmp_path):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=tmp_path)
    local_data.put_local_garden(garden_all_fields)

    garden_fields = ["uuid", "doi", "title"]
    garden_table_name = "Local Gardens"
    garden_rows = [
        (str(garden_all_fields.uuid), garden_all_fields.doi, garden_all_fields.title)
    ]
    table = Table(title=garden_table_name)

    for col in garden_fields:
        table.add_column(col)
    for row in garden_rows:
        table.add_row(*(row))

    test_table = local_data.get_local_garden_table(
        fields=garden_fields[1:], table_name=garden_table_name
    )


def test_local_storage_pipeline(
    mocker, garden_client, registered_pipeline_toy_example, tmp_path
):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=tmp_path)
    local_data.put_local_pipeline(registered_pipeline_toy_example)
    uuid = registered_pipeline_toy_example.uuid
    from_record = local_data.get_local_pipeline_by_uuid(uuid)
    assert from_record == registered_pipeline_toy_example


def test_get_local_pipeline_json(
    mocker, garden_client, registered_pipeline_toy_example, tmp_path
):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=tmp_path)
    local_data.put_local_pipeline(registered_pipeline_toy_example)

    pipeline_data = local_data._read_local_db().get(
        local_data.ResourceType.PIPELINE.value, {}
    )[str(registered_pipeline_toy_example.uuid)]

    assert pipeline_data == local_data.get_local_pipeline_json(
        str(registered_pipeline_toy_example.uuid)
    )
    assert pipeline_data == local_data.get_local_pipeline_json(
        str(registered_pipeline_toy_example.doi)
    )


def test_get_local_pipeline_table(
    mocker, garden_client, registered_pipeline_toy_example, tmp_path
):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=tmp_path)
    local_data.put_local_pipeline(registered_pipeline_toy_example)

    pipeline_fields = ["uuid", "doi", "title"]
    pipeline_table_name = "Local Pipelines"
    pipeline_rows = [
        (
            str(registered_pipeline_toy_example.uuid),
            registered_pipeline_toy_example.doi,
            registered_pipeline_toy_example.title,
        )
    ]
    table = Table(title=pipeline_table_name)

    for col in pipeline_fields:
        table.add_column(col)
    for row in pipeline_rows:
        table.add_row(*(row))

    test_table = local_data.get_local_pipeline_table(
        fields=pipeline_fields[1:], table_name=pipeline_table_name
    )
    assert table.__dict__ == test_table.__dict__


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


def test_get_local_model_json(mocker, database_with_model, second_draft_of_model):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=database_with_model)
    local_data.put_local_model(second_draft_of_model)

    model_data = local_data._read_local_db().get(
        local_data.ResourceType.MODEL.value, {}
    )[str(second_draft_of_model.model_uri)]

    assert model_data == local_data.get_local_model_json(
        str(second_draft_of_model.model_uri)
    )


def test_get_local_model_table(mocker, database_with_model, second_draft_of_model):
    # mock to replace "~/.garden/db"
    mocker.patch("garden_ai.local_data.LOCAL_STORAGE", new=database_with_model)
    local_data.put_local_model(second_draft_of_model)

    model_fields = ["model_uri", "model_name", "flavor"]
    model_table_name = "Local Models"
    model_rows = [
        (
            second_draft_of_model.model_uri,
            second_draft_of_model.model_name,
            second_draft_of_model.flavor,
        )
    ]
    model_rows = [
        (
            str(second_draft_of_model.model_uri),
            second_draft_of_model.model_name,
            second_draft_of_model.flavor,
        )
    ]
    table = Table(title=model_table_name)

    for col in model_fields:
        table.add_column(col)
    for row in model_rows:
        table.add_row(*(row))

    test_table = local_data.get_local_model_table(
        fields=model_fields[1:], table_name=model_table_name
    )
    assert table.__dict__ == test_table.__dict__
