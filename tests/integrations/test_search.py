import pytest

from garden_ai import GardenClient


@pytest.mark.integration
def test_get_pipeline_from_gsearch():
    client = GardenClient()
    doi = "10.26311/fake-doi"
    garden_id = "28b3e49c-b342-4f15-90a8-c98fe2d75105"
    garden = client.get_garden_by_id(garden_id)
    same_garden = client.get_garden_by_doi(doi)
    assert same_garden.year
    assert garden.year
