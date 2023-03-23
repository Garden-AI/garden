import pytest

from garden_ai.utils.filesystem import (
    PipelineLoadException,
    load_pipeline_from_python_file,
)
from garden_ai import Pipeline
from tests.fixtures.helpers import get_fixture_file_path  # type: ignore


def test_load_pipeline_from_valid_file():
    fixture_file_path = get_fixture_file_path("fixture_pipeline/pipeline.py")
    loaded_pipeline = load_pipeline_from_python_file(
        fixture_file_path, "fixture_pipeline"
    )
    assert isinstance(loaded_pipeline, Pipeline)


def test_load_pipeline_from_valid_file_wrong_pipeline_name():
    fixture_file_path = get_fixture_file_path("fixture_pipeline/pipeline.py")
    with pytest.raises(PipelineLoadException):
        load_pipeline_from_python_file(fixture_file_path, "does_not_match_the_file")


def test_load_pipeline_from_invalid_file():
    fixture_file_path = get_fixture_file_path("fixture_pipeline/invalid_pipeline.py")
    with pytest.raises(PipelineLoadException):
        load_pipeline_from_python_file(fixture_file_path, "fixture_pipeline")


def test_load_pipeline_from_nonexistent_pipeline():
    fixture_file_path = get_fixture_file_path("fixture_pipeline/not_actually_a_file.py")
    with pytest.raises(PipelineLoadException):
        load_pipeline_from_python_file(fixture_file_path, "fixture_pipeline")
