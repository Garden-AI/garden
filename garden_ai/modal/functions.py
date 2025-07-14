import io
from typing import TYPE_CHECKING, Any, TypeVar

import modal
import rich
from modal._serialization import deserialize, deserialize_data_format, serialize
from modal._traceback import append_modal_tb
from modal._utils.async_utils import synchronize_api
from modal._utils.blob_utils import (
    DEFAULT_SEGMENT_CHUNK_SIZE,
    MAX_OBJECT_SIZE_BYTES,
    _download_from_url,
    _upload_to_s3_url,
    get_content_length,
    perform_multipart_upload,
)
from modal._utils.bytes_io_segment_payload import BytesIOSegmentPayload
from modal._utils.hash_utils import get_upload_hashes
from modal.exception import DeserializationError, ExecutionError, RemoteError
from modal_proto import api_pb2  # type: ignore
from synchronicity.exceptions import UserCodeException  # type: ignore

if TYPE_CHECKING:
    from garden_ai.client import GardenClient
else:
    GardenClient = TypeVar("GardenClient")

from ..schemas.modal import (
    ModalBlobUploadURLRequest,
    ModalFunctionMetadata,
    ModalInvocationRequest,
    ModalInvocationResponse,
    _ModalGenericResult,
)


def _modal_deserialize(data: bytes, data_format: int | None = None) -> Any:
    """Deserialize data using Modal's deserialization functions."""
    if data_format is not None:
        return deserialize_data_format(data, data_format, None)
    return deserialize(data, None)


def _clean_and_raise_remote_exception(modal_result: api_pb2.GenericResult) -> None:
    """Handle remote exceptions from Modal, with proper traceback handling."""
    if modal_result.traceback:
        rich.print(f"[red]{modal_result.traceback}[/red]")

    if modal_result.data:
        try:
            # In these cases the data field contains an exception, not a real result
            exc = _modal_deserialize(modal_result.data)
        except DeserializationError as deser_exc:
            raise ExecutionError(
                "Could not deserialize remote exception due to local error:\n"
                + f"{deser_exc}\n"
                + "This can happen if your local environment does not have the remote exception definitions.\n"
                + "Here is the remote traceback:\n"
                + f"{modal_result.traceback}"
            ) from deser_exc.__cause__
        except Exception as deser_exc:
            raise ExecutionError(
                "Could not deserialize remote exception due to local error:\n"
                + f"{deser_exc}\n"
                + "Here is the remote traceback:\n"
                + f"{modal_result.traceback}"
            ) from deser_exc

        if not isinstance(exc, BaseException):
            raise ExecutionError(f"Got remote exception of incorrect type {type(exc)}")

        if modal_result.serialized_tb:
            try:
                tb_info: dict = _modal_deserialize(modal_result.serialized_tb)
                line_cache = _modal_deserialize(modal_result.tb_line_cache)
                append_modal_tb(exc, tb_info, line_cache)
            except:  # noqa
                # deliberately suppress exceptions -- if we can't deserialize the traceback, we still raise the original exc
                pass
        # raise synchronicity-provided exception type
        raise UserCodeException(exc)
    else:
        raise RemoteError(modal_result.exception)


class _ModalFunction:
    def __init__(
        self, metadata: ModalFunctionMetadata, client: GardenClient | None = None
    ):
        self._metadata = metadata
        self._client = client

    @property
    def metadata(self) -> ModalFunctionMetadata:
        return self._metadata

    @property
    def client(self) -> GardenClient:
        return self._client or self._get_garden_client()

    def _get_garden_client(self) -> GardenClient:
        from garden_ai import GardenClient

        return GardenClient()

    async def __call__(self, *args, **kwargs) -> Any:
        response: ModalInvocationResponse = await self._request_invocation(
            *args, **kwargs
        )
        modal_result = api_pb2.GenericResult(
            **response.result.model_dump(mode="python", exclude_defaults=True)
        )

        match modal_result.status:
            case api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
                return _modal_deserialize(modal_result.data)
            case api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT:
                raise modal.exception.FunctionTimeoutError(modal_result.exception)
            case _:
                # Not a timeout or success, but a secret third thing
                _clean_and_raise_remote_exception(modal_result)

    async def _upload_blob(self, data) -> str:
        """
        Upload large input data to Modal's blob storage through Garden, returning the data's blob ID
        """
        if isinstance(data, bytes):
            data = io.BytesIO(data)
        content_length = get_content_length(data)
        hashes = get_upload_hashes(data)

        request = ModalBlobUploadURLRequest(
            content_length=content_length,
            content_md5=hashes.md5_base64,
            content_sha256_base64=hashes.sha256_base64,
        )
        response = self.client.backend_client.get_blob_upload_url(request)

        blob_id = response.blob_id

        # perform upload using modal library helpers
        if response.upload_type == "multipart":
            assert (
                response.multipart is not None
            ), "Failed to upload large input arguments."
            await perform_multipart_upload(
                data,
                content_length=content_length,
                max_part_size=response.multipart.part_length,
                part_urls=response.multipart.upload_urls,
                completion_url=response.multipart.completion_url,
                upload_chunk_size=DEFAULT_SEGMENT_CHUNK_SIZE,
            )
        else:
            payload = BytesIOSegmentPayload(
                data,
                segment_start=0,
                segment_length=content_length,
            )
            await _upload_to_s3_url(  # type: ignore
                response.upload_url,
                payload,
                content_md5_b64=hashes.md5_base64,
            )
        return blob_id

    async def _request_invocation(self, *args, **kwargs) -> ModalInvocationResponse:
        args_kwargs_serialized = serialize((args, kwargs))

        # Handle large arguments via blob storage if needed
        if len(args_kwargs_serialized) > MAX_OBJECT_SIZE_BYTES:
            blob_id = await self._upload_blob(args_kwargs_serialized)
            request = ModalInvocationRequest(
                function_id=self.metadata.id,
                args_blob_id=blob_id,
            )
        else:
            request = ModalInvocationRequest(
                function_id=self.metadata.id,
                args_kwargs_serialized=args_kwargs_serialized,
            )

        # If we're in prod, track this invocation
        if self.client._mixpanel_track:
            event_properties = {
                "compute_type": "modal",
                "function_identifier": str(self.metadata.id),
                "function_name": self.metadata.function_name,
            }
            self.client._mixpanel_track("function_call", event_properties)

        # Invoke via backend
        response: ModalInvocationResponse = (
            self.client.backend_client.invoke_modal_function_async(request)
        )
        if (url := response.result.data_blob_url) is not None:
            # hack: make large result data look like a regular modal payload.
            blob = await _download_from_url(url)  # type: ignore
            downloaded_result = _ModalGenericResult(
                data=blob,
                **response.result.model_dump(
                    mode="python", exclude={"data", "data_blob_url"}
                ),
            )
            response.result = downloaded_result

        return response

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.metadata == other.metadata


# if you can't beat em
if TYPE_CHECKING:
    ModalFunction = _ModalFunction
else:
    ModalFunction = synchronize_api(_ModalFunction)
