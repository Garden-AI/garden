import pytest

from garden_ai.utils.filesystem import (
    PipelineLoadException,
    load_pipeline_from_python_file,
)

from garden_ai import Pipeline
from tests.fixtures.helpers import get_fixture_file_path  # type: ignore


def test_load_pipeline_from_valid_file_with_no_model():
    fixture_file_path = get_fixture_file_path("fixture_pipeline/pipeline.py")
    with pytest.raises(PipelineLoadException):
        load_pipeline_from_python_file(fixture_file_path)


def test_load_pipeline_from_valid_file_with_no_pipeline():
    fixture_file_path = get_fixture_file_path("fixture_pipeline/missing_pipeline.py")
    with pytest.raises(PipelineLoadException):
        load_pipeline_from_python_file(fixture_file_path)


def test_load_pipeline_from_valid_file_with_invalid_model():
    fixture_file_path = get_fixture_file_path(
        "fixture_pipeline/invalid_model_pipeline.py"
    )
    with pytest.raises(PipelineLoadException):
        load_pipeline_from_python_file(fixture_file_path)


def test_load_pipeline_from_invalid_file():
    fixture_file_path = get_fixture_file_path("fixture_pipeline/invalid_pipeline.py")
    with pytest.raises(SyntaxError):
        load_pipeline_from_python_file(fixture_file_path)


def test_load_pipeline_from_nonexistent_pipeline():
    fixture_file_path = get_fixture_file_path("fixture_pipeline/not_actually_a_file.py")
    with pytest.raises(FileNotFoundError):
        load_pipeline_from_python_file(fixture_file_path)


def test_incorrect_requirements_file_specification():
    fixture_file_path = get_fixture_file_path(
        "fixture_pipeline/invalid_requirements_file.py"
    )
    with pytest.raises(ValueError):
        load_pipeline_from_python_file(fixture_file_path)


def test_valid_pipeline():
    loaded_pipeline = load_pipeline_from_python_file(
        "fixtures/fixture_pipeline/valid_pipeline.py"
    )
    assert isinstance(loaded_pipeline, Pipeline)
