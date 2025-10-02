"""Basic unit tests for Garden HPC methods."""

from unittest.mock import Mock, patch

import pytest

from garden_ai.gardens import Garden, JobStatus
from garden_ai.schemas.garden import GardenMetadata


@pytest.fixture
def mock_garden():
    """Create a basic Garden instance."""
    metadata = GardenMetadata(
        doi="10.test/garden",
        title="Test Garden",
        authors=["Test Author"],
    )
    return Garden(metadata=metadata)


def test_job_status_dataclass():
    """Test JobStatus dataclass creation."""
    status = JobStatus(
        status="completed",
        results_available=True,
        stdout="some output",
        stderr="",
    )

    assert status.status == "completed"
    assert status.results_available is True
    assert status.error is None


@patch("globus_compute_sdk.Client")
def test_get_job_status_pending(mock_gc_client, mock_garden):
    """Test get_job_status returns pending status."""
    mock_client = Mock()
    mock_client.get_task.return_value = {"pending": True}
    mock_gc_client.return_value = mock_client

    status = mock_garden.get_job_status("fake-job-id")

    assert status.status == "pending"
    assert status.results_available is False


@patch("globus_compute_sdk.Client")
def test_get_job_status_completed(mock_gc_client, mock_garden):
    """Test get_job_status returns completed status."""
    mock_client = Mock()
    mock_client.get_task.return_value = {"pending": False}
    mock_client.get_result.return_value = {"some": "result"}
    mock_gc_client.return_value = mock_client

    status = mock_garden.get_job_status("fake-job-id")

    assert status.status == "completed"
    assert status.results_available is True


@patch("globus_compute_sdk.Client")
def test_get_job_status_failed(mock_gc_client, mock_garden):
    """Test get_job_status returns failed status."""
    mock_client = Mock()
    mock_client.get_task.return_value = {"pending": False}
    mock_client.get_result.return_value = {
        "error": "Something went wrong",
        "stdout": "output",
        "stderr": "error output",
    }
    mock_gc_client.return_value = mock_client

    status = mock_garden.get_job_status("fake-job-id")

    assert status.status == "failed"
    assert status.error == "Something went wrong"
    assert status.results_available is False


def test_get_results_raises_when_pending(mock_garden):
    """Test get_results raises error when job is still pending."""
    with patch.object(mock_garden, "get_job_status") as mock_status:
        mock_status.return_value = JobStatus(status="pending")

        with pytest.raises(RuntimeError, match="still pending"):
            mock_garden.get_results("fake-job-id")


def test_get_results_raises_when_failed(mock_garden):
    """Test get_results raises error when job failed."""
    with patch.object(mock_garden, "get_job_status") as mock_status:
        mock_status.return_value = JobStatus(
            status="failed",
            error="Job failed",
        )

        with pytest.raises(RuntimeError, match="failed"):
            mock_garden.get_results("fake-job-id")


@patch("globus_compute_sdk.Client")
def test_get_results_returns_result(mock_gc_client, mock_garden):
    """Test get_results returns result when job is completed."""
    mock_client = Mock()
    mock_client.get_result.return_value = "result data"
    mock_gc_client.return_value = mock_client

    with patch.object(mock_garden, "get_job_status") as mock_status:
        mock_status.return_value = JobStatus(
            status="completed",
            results_available=True,
        )

        result = mock_garden.get_results("fake-job-id")

        assert result == "result data"
