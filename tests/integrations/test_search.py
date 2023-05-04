import pytest
from uuid import UUID

from garden_ai import GardenClient


@pytest.mark.integration
def test_get_pipeline_from_gsearch():
    client = GardenClient()
    doi = "10.23677/w2ty-gv07"
    garden_id = "fcd7dca6-0622-4af5-a20d-6690418a1dae"
    garden = client.get_garden_by_id(garden_id)
    same_garden = client.get_garden_by_doi(doi)
    assert same_garden.pipeline_ids[0] == UUID("50039c98-b6c6-415a-b11b-9a0845c0a9b8")
    assert garden.pipeline_ids[0] == UUID("50039c98-b6c6-415a-b11b-9a0845c0a9b8")
