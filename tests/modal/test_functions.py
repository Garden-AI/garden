from unittest.mock import MagicMock, Mock, patch

import pytest
from garden_ai.backend_client import BackendClient
from garden_ai.client import GardenClient
from garden_ai.modal.functions import MaximumArgumentSizeError, ModalFunction
from garden_ai.schemas.modal import (
    ModalFunctionMetadata,
    ModalInvocationRequest,
    ModalInvocationResponse,
)


@pytest.fixture
def modal_function_meta(modal_function_metadata_json):
    return ModalFunctionMetadata(**modal_function_metadata_json)


@pytest.fixture
def modal_function(modal_function_meta):
    mock_garden_client = Mock(spec=GardenClient)
    mock_backend_client = Mock(spec=BackendClient)
    mock_garden_client.backend_client = mock_backend_client
    mock_garden_client._mixpanel_track = Mock()
    return ModalFunction(metadata=modal_function_meta, client=mock_garden_client)


def test_modal_function_call(modal_function):
    # Mocks for serializing args
    mock_args_kwargs = (("fake input data",), {"hello": "world"})
    mock_args_kwargs_serialized = b"careful! I byte"
    mock_serialize = Mock(return_value=mock_args_kwargs_serialized)

    # MagicMock for the modal.Client.anonymous context manager
    mock_modal_client = MagicMock()
    mock_modal_client_instance = Mock()
    mock_modal_client_instance.stub = Mock()
    mock_modal_client.return_value.__enter__.return_value = mock_modal_client_instance

    # Mock for the async process helper
    mock_modal_process_result_helper = Mock(
        return_value="what results! and *so* nicely processed"
    )

    # Mock the backend client's invoke_modal_function method
    modal_function.client.backend_client.invoke_modal_function_async.return_value = (
        ModalInvocationResponse(
            result=dict(status=0, data=b"mock unprocessed data"), data_format=1
        )
    )

    with (
        patch("garden_ai.modal.functions.serialize", mock_serialize),
        patch(
            "garden_ai.modal.functions._modal_process_result_sync",
            mock_modal_process_result_helper,
        ),
        patch("modal.Client.anonymous", mock_modal_client),
    ):
        # call the modal function object
        result = modal_function("fake input data", hello="world")

    # Check that serialize was called with the correct arguments
    mock_serialize.assert_called_once_with(mock_args_kwargs)
    # Check that the backend client's method was called with the correct request payload
    modal_function.client.backend_client.invoke_modal_function_async.assert_called_once_with(
        ModalInvocationRequest(
            function_id=42,
            args_kwargs_serialized=mock_args_kwargs_serialized,
        )
    )

    # Check that modal.Client.anonymous was called with the correct URL
    mock_modal_client.assert_called_once_with("https://api.modal.com")

    # Check that _modal_process_result_sync was called
    mock_modal_process_result_helper.assert_called_once()

    # Check that the function invocation tracker was called
    modal_function.client._mixpanel_track.assert_called_once()

    # Check that the result was processed correctly
    assert result == "what results! and *so* nicely processed"


def test_modal_function_size_limit(modal_function):
    scary_input = b"oo" * (1024 * 1024 + 1)  # just over 2MiB
    mock_serialize = Mock(return_value=scary_input)

    with patch("garden_ai.modal.functions.serialize", mock_serialize):
        with pytest.raises(MaximumArgumentSizeError):
            modal_function("👻")
