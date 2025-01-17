from unittest.mock import AsyncMock, Mock, patch

import pytest

from garden_ai.backend_client import BackendClient
from garden_ai.client import GardenClient
from garden_ai.modal.functions import MAX_OBJECT_SIZE_BYTES, ModalFunction
from garden_ai.schemas.modal import (
    ModalFunctionMetadata,
    ModalInvocationRequest,
    ModalInvocationResponse,
    _ModalGenericResult,
)


@pytest.fixture
def modal_function_meta(modal_function_metadata_json):
    return ModalFunctionMetadata(**modal_function_metadata_json)


@pytest.fixture
def modal_function(modal_function_meta):
    mock_garden_client = Mock(spec=GardenClient)
    mock_backend_client = Mock(spec=BackendClient)
    mock_garden_client.backend_client = mock_backend_client
    mock_garden_client._mixpanel_track = None
    return ModalFunction(metadata=modal_function_meta, client=mock_garden_client)


def test_modal_function_call_direct(modal_function):
    """Test normal function call with arguments under size limit"""
    # Mocks for serializing args
    mock_args_kwargs = (("fake input data",), {"hello": "world"})
    mock_args_kwargs_serialized = b"careful! I byte"
    mock_result_serialized = b"mock unprocessed data"
    mock_result_deserialized = "what results! and *so* nicely processed"

    mock_serialize = Mock(return_value=mock_args_kwargs_serialized)
    mock_deserialize = AsyncMock(return_value=mock_result_deserialized)

    # Mock the backend client's invoke_modal_function_async method
    modal_function.client.backend_client.invoke_modal_function_async = Mock(
        return_value=ModalInvocationResponse(
            result=_ModalGenericResult(status=0, data=mock_result_serialized),
            data_format=1,
        )
    )

    with (
        patch("garden_ai.modal.functions.serialize", mock_serialize),
        patch("garden_ai.modal.functions._process_result", mock_deserialize),
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

    # Check that process_result was called
    mock_deserialize.assert_called_once()

    # Check that the result was processed correctly
    assert result == "what results! and *so* nicely processed"


def test_modal_function_call_large_args(modal_function):
    """Test function call with arguments over size limit that need blob storage"""
    # Mock large serialized args that exceed MAX_OBJECT_SIZE_BYTES
    mock_args_kwargs_serialized = b"x" * (MAX_OBJECT_SIZE_BYTES + 1)
    mock_result_serialized = b"mock unprocessed data"
    mock_result_deserialized = "processed blob result"
    mock_blob_id = "test-blob-123"

    # Mock the upload URL response
    mock_upload_url_response = Mock(
        blob_id=mock_blob_id,
        upload_type="single",
        upload_url="https://test-upload-url",
    )
    modal_function.client.backend_client.get_blob_upload_url = Mock(
        return_value=mock_upload_url_response
    )

    # Mock blob upload helpers
    mock_get_content_length = Mock(return_value=len(mock_args_kwargs_serialized))
    mock_upload_to_s3 = AsyncMock()

    # Mock response from function invocation
    modal_function.client.backend_client.invoke_modal_function_async = Mock(
        return_value=ModalInvocationResponse(
            result=_ModalGenericResult(status=0, data=mock_result_serialized),
            data_format=1,
        )
    )

    with (
        patch(
            "garden_ai.modal.functions.serialize",
            return_value=mock_args_kwargs_serialized,
        ),
        patch("garden_ai.modal.functions.get_content_length", mock_get_content_length),
        patch("garden_ai.modal.functions._upload_to_s3_url", mock_upload_to_s3),
        patch(
            "garden_ai.modal.functions._process_result",
            AsyncMock(return_value=mock_result_deserialized),
        ),
    ):
        result = modal_function("large fake input data", hello="world")

    # Verify request was made with blob_id instead of direct args
    modal_function.client.backend_client.invoke_modal_function_async.assert_called_once_with(
        ModalInvocationRequest(
            function_id=42,
            args_blob_id=mock_blob_id,
        )
    )

    # Verify upload URL was requested and blob was uploaded
    modal_function.client.backend_client.get_blob_upload_url.assert_called_once()
    mock_upload_to_s3.assert_called_once()
    assert result == mock_result_deserialized


def test_modal_function_call_large_args_multipart(modal_function):
    """Test function call with arguments so large they need multipart upload"""
    mock_args_kwargs_serialized = b"x" * (MAX_OBJECT_SIZE_BYTES * 2)
    mock_result_serialized = b"mock unprocessed data"
    mock_result_deserialized = "processed multipart blob result"
    mock_blob_id = "test-blob-456"

    # Mock multipart upload URL response
    mock_upload_url_response = Mock(
        blob_id=mock_blob_id,
        upload_type="multipart",
        multipart=Mock(
            part_length=MAX_OBJECT_SIZE_BYTES // 2,
            upload_urls=["https://test-upload-url-1", "https://test-upload-url-2"],
            completion_url="https://test-completion-url",
        ),
    )
    modal_function.client.backend_client.get_blob_upload_url = Mock(
        return_value=mock_upload_url_response
    )

    # Mock blob upload helpers
    mock_get_content_length = Mock(return_value=len(mock_args_kwargs_serialized))
    mock_perform_multipart = AsyncMock()

    # Mock response from function invocation
    modal_function.client.backend_client.invoke_modal_function_async = Mock(
        return_value=ModalInvocationResponse(
            result=_ModalGenericResult(status=0, data=mock_result_serialized),
            data_format=1,
        )
    )

    with (
        patch(
            "garden_ai.modal.functions.serialize",
            return_value=mock_args_kwargs_serialized,
        ),
        patch("garden_ai.modal.functions.get_content_length", mock_get_content_length),
        patch(
            "garden_ai.modal.functions.perform_multipart_upload", mock_perform_multipart
        ),
        patch(
            "garden_ai.modal.functions._process_result",
            AsyncMock(return_value=mock_result_deserialized),
        ),
    ):
        result = modal_function("very large fake input data", hello="world")

    # Verify request was made with blob_id
    modal_function.client.backend_client.invoke_modal_function_async.assert_called_once_with(
        ModalInvocationRequest(
            function_id=42,
            args_blob_id=mock_blob_id,
        )
    )

    # Verify multipart upload was initiated and completed
    modal_function.client.backend_client.get_blob_upload_url.assert_called_once()
    mock_perform_multipart.assert_called_once()
    assert result == mock_result_deserialized


def test_modal_function_call_blob_response(modal_function):
    """Test function call that receives result via blob storage"""
    mock_args_kwargs_serialized = b"small input"
    mock_blob_data = b"large result from blob storage"
    mock_result_deserialized = "processed blob result"
    mock_download_url = "https://test-download-url"

    # Mock the download response
    mock_download = AsyncMock(return_value=mock_blob_data)

    # Mock response from function invocation with blob URL
    modal_function.client.backend_client.invoke_modal_function_async = Mock(
        return_value=ModalInvocationResponse(
            result=_ModalGenericResult(
                status=0,
                data_blob_url=mock_download_url,
            ),
            data_format=1,
        )
    )

    with (
        patch(
            "garden_ai.modal.functions.serialize",
            return_value=mock_args_kwargs_serialized,
        ),
        patch("garden_ai.modal.functions._download_from_url", mock_download),
        patch(
            "garden_ai.modal.functions._process_result",
            AsyncMock(return_value=mock_result_deserialized),
        ),
    ):
        result = modal_function("input data")

    # Verify blob was downloaded
    mock_download.assert_called_once_with(mock_download_url)
    assert result == mock_result_deserialized
