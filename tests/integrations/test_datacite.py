import pytest


@pytest.mark.integration
def test_auto_doi_garden(garden_no_fields):
    garden = garden_no_fields
    assert not garden.doi
    garden.authors = ["pytest"]
    garden.title = "DOI request test (Garden)"
    garden.request_doi()
    assert garden.doi


@pytest.mark.integration
def test_auto_doi_pipelines(pipeline_toy_example):
    pipe = pipeline_toy_example
    assert not pipe.doi
    pipe.authors = ["pytest"]
    pipe.title = "DOI request test (Pipeline)"
    pipe.request_doi()
    assert pipe.doi
