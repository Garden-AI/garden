import pytest

from garden_ai import GardenClient


@pytest.mark.integration
def test_get_pipeline_from_gsearch():
    client = GardenClient()
    doi = "10.23677/w2ty-gv07"
    same_garden = client.get_published_garden(doi)
    assert same_garden.contributors == ["Ward, Logan", "The Accelerate Gang", "et al."]
