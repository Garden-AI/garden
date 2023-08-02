from garden_ai.app.completion import complete_model, complete_pipeline, complete_garden


def test_complete_model(second_draft_of_model, mocker):
    registered_model = second_draft_of_model
    mocker.patch("garden_ai.local_data.get_all_local_models").return_value = [
        registered_model
    ]

    model_full_name = registered_model.full_name

    empty_search = complete_model("")
    assert [model_full_name] == empty_search

    not_existing_search = complete_model("not-a-full-model-name-that-can-be-found")
    assert len(not_existing_search) == 0


def test_complete_pipeline(database_with_connected_pipeline, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_pipeline
    )
    pipeline_doi = "10.23677/jx31-gx98"
    pipeline_title = "Fixture pipeline"

    empty_search = complete_pipeline("")
    assert [(pipeline_doi, pipeline_title)] == empty_search

    not_existing_search = complete_model("a-pipeline-that-cannot-be-found")
    assert len(not_existing_search) == 0


def test_complete_garden(database_with_connected_pipeline, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_pipeline
    )
    garden_doi = "10.23677/fake-doi"
    garden_title = "Will Test Garden"

    empty_search = complete_garden("")
    assert [(garden_doi, garden_title)] == empty_search

    not_existing_search = complete_model("a-garden-that-cannot-be-found")
    assert len(not_existing_search) == 0
