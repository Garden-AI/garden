"""Basic unit tests for HpcFunction."""

import pytest

from garden_ai.hpc.functions import HpcFunction
from garden_ai.schemas.hpc import HpcDeploymentInfo, HpcFunctionMetadata


@pytest.fixture
def sample_metadata():
    """Create sample HPC function metadata."""
    return HpcFunctionMetadata(
        id=123,
        function_name="my_hpc_function",
        function_text="def my_hpc_function(x):\n    return x * 2",
        title="My HPC Function",
        available_deployments=[
            HpcDeploymentInfo(
                deployment_id=1,
                endpoint_name="edith",
                endpoint_gcmu_id="abc-123",
                conda_env_path="/path/to/env1",
            ),
            HpcDeploymentInfo(
                deployment_id=2,
                endpoint_name="polaris",
                endpoint_gcmu_id="def-456",
                conda_env_path="/path/to/env2",
            ),
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

    with pytest.raises(NotImplementedError, match="must be submitted asynchronously"):
        func("some_arg")


def test_hpc_function_submit_requires_endpoint_id(sample_metadata):
    """Test that submit() requires endpoint_id parameter."""
    func = HpcFunction(sample_metadata)

    with pytest.raises(ValueError, match="Must provide endpoint_id"):
        func.submit(arg1="value")


def test_hpc_function_submit_validates_endpoint_id(sample_metadata):
    """Test that submit() validates endpoint_id exists in deployments."""
    func = HpcFunction(sample_metadata)

    with pytest.raises(ValueError, match="No deployment found for endpoint"):
        func.submit(endpoint_id="invalid-endpoint", arg1="value")


def test_hpc_function_deduplicates_endpoints():
    """Test that duplicate endpoints are deduplicated."""
    metadata = HpcFunctionMetadata(
        id=123,
        function_name="test_func",
        function_text="def test_func(): pass",
        available_deployments=[
            HpcDeploymentInfo(
                deployment_id=1,
                endpoint_name="edith",
                endpoint_gcmu_id="abc-123",
                conda_env_path="/path/1",
            ),
            HpcDeploymentInfo(
                deployment_id=2,
                endpoint_name="edith",  # Same endpoint, different deployment
                endpoint_gcmu_id="abc-123",
                conda_env_path="/path/2",
            ),
        ],
    )

    func = HpcFunction(metadata)

    # Should only have one endpoint despite two deployments
    assert len(func.endpoints) == 1
    assert func.endpoints[0]["id"] == "abc-123"
