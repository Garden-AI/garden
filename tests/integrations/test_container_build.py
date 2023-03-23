import pytest

from garden_ai import GardenClient


@pytest.mark.integration
def test_container_build(pipeline_toy_example):
    gc = GardenClient()
    gc.build_container(pipeline_toy_example)
