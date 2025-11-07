"""Basic unit tests for HpcFunction."""

import pytest

from garden_ai.hpc.functions import HpcFunction
from garden_ai.schemas.hpc import HpcFunctionMetadata

sample_groundhog_source = """
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "torch",
# ]
#
# ///
import os

import groundhog_hpc as hog

@hog.function(walltime=30, account="cis250223")
def my_hpc_function():
    return dict(os.environ)

@hog.harness()
def main():
    print("running remotely...")
    remote_env = my_hpc_function.remote()
    print(json.dumps(remote_env, indent=2))
    return remote_env
"""


@pytest.fixture
def sample_metadata():
    """Create sample HPC function metadata."""
    return HpcFunctionMetadata(
        id=123,
        function_name="my_hpc_function",
        function_text=sample_groundhog_source,
        title="My HPC Function",
        available_endpoints=[
            {"id": "abc-123", "name": "edith"},
            {"id": "def-456", "name": "polaris"},
        ],
    )


def test_hpc_function_initialization(sample_metadata):
    """Test that HpcFunction initializes correctly."""
    func = HpcFunction(sample_metadata)

    assert func.metadata == sample_metadata
    assert len(func.endpoints) == 2


def test_hpc_function_endpoints_extraction(sample_metadata):
    """Test that endpoints are extracted correctly from deployments."""
    func = HpcFunction(sample_metadata)

    endpoint_ids = [e["id"] for e in func.endpoints]
    endpoint_names = [e["name"] for e in func.endpoints]

    assert "abc-123" in endpoint_ids
    assert "def-456" in endpoint_ids
    assert "edith" in endpoint_names
    assert "polaris" in endpoint_names


def test_hpc_function_call_raises_error(sample_metadata):
    """Test that calling HpcFunction directly raises NotImplementedError."""
    func = HpcFunction(sample_metadata)

    with pytest.raises(
        NotImplementedError, match="HPC functions cannot be called directly"
    ):
        func("some_arg")
