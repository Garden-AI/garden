import pytest


@pytest.mark.integration
def test_auto_doi_garden(garden_client, garden_no_fields):
    garden = garden_no_fields
    assert not garden.doi
    garden.authors = ["pytest"]
    garden.title = "DOI request test (Garden)"
    garden.doi = garden_client._mint_doi(garden)
    assert garden.doi


@pytest.mark.integration
def test_auto_doi_pipelines(garden_client, pipeline_toy_example):
    pipe = pipeline_toy_example
    assert not pipe.doi
    pipe.authors = ["pytest"]
    pipe.title = "DOI request test (Pipeline)"
    garden_client._mint_doi(pipe)
    assert pipe.doi
