import pytest

from garden_ai.app.pipeline import register
from tests.fixtures.helpers import get_fixture_file_path  # type: ignore


@pytest.mark.integration
def test_register_pipeline():
    fixture_pipeline_path = get_fixture_file_path("fixture_pipeline/pipeline.py")
    register(fixture_pipeline_path, "fixture_pipeline")
