from garden_ai.app.completion import complete_entrypoint, complete_garden


def test_complete_entrypoint(database_with_connected_entrypoint, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_entrypoint
    )
    entrypoint_doi = "10.23677/jx31-gx98"
    entrypoint_title = "Fixture entrypoint"

    empty_search = complete_entrypoint("")
    assert [(entrypoint_doi, entrypoint_title)] == empty_search


def test_complete_garden(database_with_connected_entrypoint, mocker):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_entrypoint
    )
    garden_doi = "10.23677/fake-doi"
    garden_title = "Will Test Garden"

    empty_search = complete_garden("")
    assert [(garden_doi, garden_title)] == empty_search
