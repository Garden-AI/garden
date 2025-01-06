import io
from typing import TYPE_CHECKING, TypeVar

import modal
from modal._serialization import serialize
from modal._utils.async_utils import synchronize_api
from modal._utils.blob_utils import (
    DEFAULT_SEGMENT_CHUNK_SIZE,
    MAX_OBJECT_SIZE_BYTES,
    BytesIOSegmentPayload,
    _download_from_url,
    _upload_to_s3_url,
    get_content_length,
    perform_multipart_upload,
)
from modal._utils.function_utils import _process_result
from modal._utils.hash_utils import get_upload_hashes
from modal_proto import api_pb2  # type: ignore

if TYPE_CHECKING:
    from garden_ai.client import GardenClient
else:
    GardenClient = TypeVar("GardenClient")

from ..schemas.modal import (
    ModalBlobUploadURLRequest,
    ModalFunctionMetadata,
    ModalInvocationRequest,
    ModalInvocationResponse,
)


class _ModalFunction:
    def __init__(
        self, metadata: ModalFunctionMetadata, client: GardenClient | None = None
    ):
        self._metadata = metadata
        if client:
            self.client = client
        else:
            self.client = self._get_garden_client()

    @property
    def metadata(self) -> ModalFunctionMetadata:
        return self._metadata

    def _get_garden_client(self) -> GardenClient:
        from garden_ai import GardenClient

        return GardenClient()

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
            await _upload_to_s3_url(
                response.upload_url,
                payload,
                content_md5_b64=hashes.md5_base64,
            )
        return blob_id

    async def __call__(self, *args, **kwargs):
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

        response: ModalInvocationResponse = (
            self.client.backend_client.invoke_modal_function_async(request)
        )
        result_data: dict = response.result.model_dump(
            mode="python", exclude_defaults=True
        )

        if (url := result_data.get("data_blob_url")) is not None:
            # hack: make it look like a regular payload for modal to deserialize
            blob = await _download_from_url(url)
            result_data["data"] = blob
            del result_data["data_blob_url"]

        modal_result_struct = api_pb2.GenericResult(**result_data)

        # If we're in prod, track this invocation
        if self.client._mixpanel_track:
            event_properties = {
                "compute_type": "modal",
                "function_identifier": str(self.metadata.id),
                "function_name": self.metadata.function_name,
            }
            self.client._mixpanel_track("function_call", event_properties)

        async with modal.Client.anonymous("https://api.modal.com") as client:
            return await _process_result(
                modal_result_struct, response.data_format, client.stub, client
            )

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.metadata == other.metadata


# if you can't beat em
ModalFunction = synchronize_api(_ModalFunction)
