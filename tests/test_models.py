import json

from garden_ai import (
    PublishedGarden,
    RegisteredPipeline,
    local_data,
)


def test_create_empty_garden(garden_client):
    # feels silly, but we do want users to be able to initialize an empty garden
    # & fill in required stuff later

    # object should exist with default-illegal fields
    garden = garden_client.create_garden()

    assert not garden.authors
    assert not garden.title


def test_garden_datacite(garden_title_authors_doi_only):
    data = json.loads(
        PublishedGarden.from_garden(garden_title_authors_doi_only).datacite_json()
    )

    assert isinstance(data["creators"], list)
    assert isinstance(data["titles"], list)
    assert data["publisher"] == "thegardens.ai"


def test_pipeline_datacite(registered_pipeline_toy_example):
    data = json.loads(registered_pipeline_toy_example.datacite_json())

    assert isinstance(data["creators"], list)
    assert isinstance(data["titles"], list)
    assert data["publisher"] == "thegardens.ai"


def test_garden_can_access_pipeline_as_attribute(
    mocker, database_with_connected_pipeline
):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_pipeline
    )
    garden = local_data.get_local_garden_by_doi("10.23677/fake-doi")
    published = PublishedGarden.from_garden(garden)
    assert isinstance(published.fixture_pipeline, RegisteredPipeline)
