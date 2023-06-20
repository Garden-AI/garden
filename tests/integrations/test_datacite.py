import pytest


@pytest.mark.integration
def test_auto_doi_garden(garden_client, garden_no_fields):
    garden = garden_no_fields
    garden.authors = ["pytest"]
    garden.title = "DOI request test (Garden)"
    garden.doi = garden_client._mint_draft_doi()
    assert garden.doi


@pytest.mark.integration
def test_auto_doi_pipelines(garden_client, pipeline_toy_example):
    pipe = pipeline_toy_example
    pipe.authors = ["pytest"]
    pipe.title = "DOI request test (Pipeline)"
    pipe.doi = garden_client._mint_draft_doi()
    assert pipe.doi
