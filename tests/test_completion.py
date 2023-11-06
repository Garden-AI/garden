from garden_ai.app.completion import complete_pipeline, complete_garden


def test_complete_pipeline(database_with_connected_pipeline, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_pipeline
    )
    pipeline_doi = "10.23677/jx31-gx98"
    pipeline_title = "Fixture pipeline"

    empty_search = complete_pipeline("")
    assert [(pipeline_doi, pipeline_title)] == empty_search


def test_complete_garden(database_with_connected_pipeline, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_pipeline
    )
    garden_doi = "10.23677/fake-doi"
    garden_title = "Will Test Garden"

    empty_search = complete_garden("")
    assert [(garden_doi, garden_title)] == empty_search
