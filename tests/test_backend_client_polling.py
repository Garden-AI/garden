from unittest.mock import Mock, call

import pytest
import requests

from garden_ai.constants import GardenConstants
from garden_ai.schemas.modal import ModalInvocationRequest, ModalInvocationResponse


def test_invoke_modal_function_async_basic_success(mocker, backend_client):
    """Test standard success path with no retries or backoff."""
    # Setup
    payload = ModalInvocationRequest(function_id=1, args_kwargs_serialized=b"abc")

    # Mock post response (initial invocation)
    mocker.patch.object(backend_client, "_post", return_value={"id": "job-123"})

    # Mock get response (polling)
    # First pending, then done
    mocker.patch.object(
        backend_client,
        "_get",
        side_effect=[
            {"status": "pending", "id": "job-123"},
            {
                "status": "done",
                "result": {"status": 1, "data": b"success"},
                "id": "job-123",
            },
        ],
    )

    mock_sleep = mocker.patch("time.sleep")

    # Execute
    response = backend_client.invoke_modal_function_async(payload)

    # Verify
    assert isinstance(response, ModalInvocationResponse)
    assert response.result.data == b"success"
    # Should have slept once (standard interval)
    mock_sleep.assert_called_once_with(GardenConstants.BACKEND_POLL_INTERVAL_SECONDS)


def test_invoke_modal_function_async_backoff(mocker, backend_client):
    """Test that polling slows down after 30 seconds."""
    payload = ModalInvocationRequest(function_id=1, args_kwargs_serialized=b"abc")
    mocker.patch.object(backend_client, "_post", return_value={"id": "job-123"})

    # We want to simulate > 30 seconds elapsing.
    # We'll mock time.time to return a sequence of values.
    start_time = 1000.0
    # Steps: start, check 1 (pending, elapsed=0), check 2 (pending, elapsed=31), check 3 (done)
    mocker.patch(
        "time.time",
        side_effect=[
            start_time,  # Initial start time capture
            start_time + 1.0,  # 1st loop check
            start_time + 32.0,  # 2nd loop check (>30s)
            start_time + 35.0,  # 3rd loop check
        ],
    )

    mocker.patch.object(
        backend_client,
        "_get",
        side_effect=[
            {"status": "pending", "id": "job-123"},  # call 1 triggers sleep(0.1)
            {"status": "pending", "id": "job-123"},  # call 2 triggers sleep(5.0)
            {
                "status": "done",
                "result": {"status": 1, "data": b"success"},
                "id": "job-123",
            },
        ],
    )

    mock_sleep = mocker.patch("time.sleep")

    backend_client.invoke_modal_function_async(payload)

    # Check sleep calls
    # 1. First iteration: elapsed 1.0 < 30 -> sleep default (0.1)
    # 2. Second iteration: elapsed 32.0 > 30 -> sleep 5.0
    calls = mock_sleep.call_args_list
    assert len(calls) == 2
    assert calls[0] == call(GardenConstants.BACKEND_POLL_INTERVAL_SECONDS), (
        "First sleep should be fast"
    )
    assert calls[1] == call(5.0), "Second sleep should be slow backoff"


def test_invoke_modal_function_async_502_retry_success(mocker, backend_client):
    """Test that 502 errors are retried and succeed."""
    payload = ModalInvocationRequest(function_id=1, args_kwargs_serialized=b"abc")
    mocker.patch.object(backend_client, "_post", return_value={"id": "job-123"})

    # Mock 502 Bad Gateway response
    bad_gateway = requests.HTTPError()
    bad_gateway.response = Mock(status_code=502)

    # Sequence:
    # 1. Pending
    # 2. 502 Error (Retry 1)
    # 3. 502 Error (Retry 2)
    # 4. Done
    mocker.patch.object(
        backend_client,
        "_get",
        side_effect=[
            {"status": "pending", "id": "job-123"},
            bad_gateway,
            bad_gateway,
            {
                "status": "done",
                "result": {"status": 1, "data": b"success"},
                "id": "job-123",
            },
        ],
    )

    mock_sleep = mocker.patch("time.sleep")
    mocker.patch(
        "time.time", return_value=1000.0
    )  # Always early, so short poll interval

    backend_client.invoke_modal_function_async(payload)

    # Calls verification:
    # 1. pending -> sleep(0.1) -> get
    # 2. 502 -> sleep(1.0) -> continue -> sleep(0.1) -> get (Actually loop structure might vary)
    # Based on planned implementation:
    # Loop top: status=pending
    # Sleep 0.1
    # Try get -> Success pending (loop continues)
    # Loop top: status=pending
    # Sleep 0.1
    # Try get -> 502 -> catch -> retries++ -> sleep(1.0) -> continue
    # Loop top: status=pending
    # Sleep 0.1
    # Try get -> 502 -> catch -> retries++ -> sleep(1.0) -> continue
    # Loop top: status=pending
    # Sleep 0.1
    # Try get -> Done

    # Total sleeps:
    # 0.1 (before first get)
    # 0.1 (before second get, which fails)
    # 1.0 (inside 502 handler)
    # 0.1 (before third get, which fails)
    # 1.0 (inside 502 handler)
    # 0.1 (before fourth get, which succeeds)

    # Verify specific calls to ensure retry logic triggered
    assert call(1.0) in mock_sleep.call_args_list, "Should sleep 1.0s on 502"
    assert mock_sleep.call_count >= 5


def test_invoke_modal_function_async_502_retry_failure(mocker, backend_client):
    """Test that 502 errors eventually raise after max retries."""
    payload = ModalInvocationRequest(function_id=1, args_kwargs_serialized=b"abc")
    mocker.patch.object(backend_client, "_post", return_value={"id": "job-123"})

    bad_gateway = requests.HTTPError()
    bad_gateway.response = Mock(status_code=502)

    # Fail 5 times (limit is 3 retries, so total 4 attempts)
    mocker.patch.object(backend_client, "_get", side_effect=bad_gateway)

    mocker.patch("time.sleep")
    mocker.patch("time.time", return_value=1000.0)

    with pytest.raises(requests.HTTPError):
        backend_client.invoke_modal_function_async(payload)


def test_invoke_modal_function_async_non_502_error(mocker, backend_client):
    """Test that other errors (e.g. 404) are raised immediately."""
    payload = ModalInvocationRequest(function_id=1, args_kwargs_serialized=b"abc")
    mocker.patch.object(backend_client, "_post", return_value={"id": "job-123"})

    not_found = requests.HTTPError()
    not_found.response = Mock(status_code=404)

    mocker.patch.object(backend_client, "_get", side_effect=not_found)

    with pytest.raises(requests.HTTPError):
        backend_client.invoke_modal_function_async(payload)
