import unittest
from unittest.mock import patch, MagicMock
from garden_ai import EntrypointMetadata, garden_entrypoint, entrypoint_test
from garden_ai.entrypoints import (
    RegisteredEntrypoint,
    EntrypointIdempotencyError,
)  # Adjust import paths as necessary

import numpy as np
import pandas as pd  # type: ignore

# Mock UUIDs for testing
function_uuid = "123e4567-e89b-12d3-a456-426614174000"
container_uuid = "123e4567-e89b-12d3-a456-426614174001"


@patch("garden_ai.entrypoints.globus_compute_sdk.Executor")
def test_dlhub_entrypoint(mock_executor):
    dlhub_entrypoint = RegisteredEntrypoint(
        doi="10.26311/3hz8-as26",  # A DOI in the DLHub list
        title="Migrated DLHub Model",
        short_name="run_dlhub_model",
        authors=["Joe Schmoe"],
        func_uuid=function_uuid,
        container_uuid=function_uuid,
    )

    dlhub_wrapped_result = (("answer", {"stdout": None, "success": True}), 3452354)

    mock_executor_instance = MagicMock()
    mock_executor.return_value.__enter__.return_value = mock_executor_instance
    mock_executor_instance.submit_to_registered_function.return_value.result.return_value = (
        dlhub_wrapped_result
    )

    result = dlhub_entrypoint("test arg", endpoint="specific-endpoint")

    assert (
        result == "answer"
    ), "Should return the direct result for an entrypoint with DOI in list"
    mock_executor_instance.submit_to_registered_function.assert_called_once()
    mock_executor_instance.submit_to_registered_function.assert_called_with(
        args=(
            {
                "inputs": "test arg",
                "parameters": [],
                "debug": False,
            },
        ),
        function_id=function_uuid,
        kwargs={},
    )


@patch("garden_ai.entrypoints.globus_compute_sdk.Executor")
def test_normal_entrypoint(mock_executor):
    normal_entrypoint = RegisteredEntrypoint(
        doi="foo",  # A DOI not in the DLHub list
        title="Some cool model",
        short_name="run_model",
        authors=["Jane Schmane"],
        func_uuid=function_uuid,
        container_uuid=function_uuid,
    )

    mock_executor_instance = MagicMock()
    mock_executor.return_value.__enter__.return_value = mock_executor_instance
    mock_executor_instance.submit_to_registered_function.return_value.result.return_value = (
        "mocked result"
    )

    result = normal_entrypoint("test arg", endpoint="specific-endpoint")

    assert (
        result == "mocked result"
    ), "Should return the direct result for an entrypoint with DOI not in list"
    mock_executor_instance.submit_to_registered_function.assert_called_once()


def test_non_idempotent_garden_entrypoint_raises_error():
    class NonIdempotentCounter:
        def __init__(self):
            self.times_called = 0

        def increment(self):
            self.times_called += 1
            return self.times_called

    counter = NonIdempotentCounter()

    metadata = EntrypointMetadata(
        title="Fake Entrypoint",
        description="A sample description",
        authors=["Farnsworth, Hubert J."],
        tags=["test"],
    )

    # Setup a mock entrypoint that is non-idempotent
    @garden_entrypoint(metadata=metadata)
    def simple_non_idempotent_entrypoint_func():
        return counter.increment()

    # Setup a simple entrypoint_test
    @entrypoint_test(simple_non_idempotent_entrypoint_func)
    def simple_test_the_entrypoint():
        result = simple_non_idempotent_entrypoint_func()
        return result

    # Assert the entrypoint test throws an error due to non-idempotency
    with unittest.TestCase().assertRaises(EntrypointIdempotencyError):
        simple_test_the_entrypoint()

    # Setup a mock entrypoint that is non-idempotent with ndarray return
    @garden_entrypoint(metadata=metadata)
    def numpy_non_idempotent_entrypoint_func():
        n = counter.increment()
        return np.array([n, n, n])

    # Setup numpy entrypoint_test
    @entrypoint_test(numpy_non_idempotent_entrypoint_func)
    def numpy_test_the_entrypoint():
        result = numpy_non_idempotent_entrypoint_func()
        return result

    # Assert the entrypoint test throws an error due to non-idempotency
    with unittest.TestCase().assertRaises(EntrypointIdempotencyError):
        numpy_test_the_entrypoint()

    # Setup a mock entrypoint that is non-idempotent with dataframe return
    @garden_entrypoint(metadata=metadata)
    def pandas_non_idempotent_entrypoint_func():
        n = counter.increment()
        return pd.DataFrame([n, n, n])

    # Setup pandas entrypoint_test
    @entrypoint_test(pandas_non_idempotent_entrypoint_func)
    def pandas_test_the_entrypoint():
        result = pandas_non_idempotent_entrypoint_func()
        return result

    # Assert the entrypoint test throws an error due to non-idempotency
    with unittest.TestCase().assertRaises(EntrypointIdempotencyError):
        pandas_test_the_entrypoint()


def test_idempotent_garden_entrpoint_passes():
    metadata = EntrypointMetadata(
        title="Fake Entrypoint",
        description="A sample description",
        authors=["Farnsworth, Hubert J."],
        tags=["test"],
    )

    # Setup a mock entrypoint that is idempotent
    @garden_entrypoint(metadata=metadata)
    def idempotent_entrypoint_func():
        return True

    # Setup a simple entrypoint_test
    @entrypoint_test(idempotent_entrypoint_func)
    def test_the_entrypoint():
        result = idempotent_entrypoint_func()
        return result

    # Assert the test returns the value as it should pass the entrypoint_test
    assert test_the_entrypoint() is True

    # Setup a mock entrypoint that is idempotent with ndarray return
    @garden_entrypoint(metadata=metadata)
    def numpy_idempotent_entrypoint_func():
        return np.array([1, 2, 3])

    # Setup numpy entrypoint_test
    @entrypoint_test(numpy_idempotent_entrypoint_func)
    def numpy_test_the_entrypoint():
        result = numpy_idempotent_entrypoint_func()
        return result

    # Assert the test returns the value as it should pass the entrypoint_test
    assert np.array_equal(numpy_test_the_entrypoint(), np.array([1, 2, 3])) is True

    # Setup a mock entrypoint that is idempotent with dataframe return
    @garden_entrypoint(metadata=metadata)
    def pandas_idempotent_entrypoint_func():
        return pd.DataFrame([1, 2, 3])

    # Setup pandas entrypoint_test
    @entrypoint_test(pandas_idempotent_entrypoint_func)
    def pandas_test_the_entrypoint():
        result = pandas_idempotent_entrypoint_func()
        return result

    # Assert the test returns the value as it should pass the entrypoint_test
    assert pandas_test_the_entrypoint().equals(pd.DataFrame([1, 2, 3])) is True
