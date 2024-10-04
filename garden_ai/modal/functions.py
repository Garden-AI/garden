import asyncio
from typing import Any, TYPE_CHECKING, TypeVar

import modal
from modal._serialization import serialize
from modal._utils.blob_utils import MAX_OBJECT_SIZE_BYTES
from modal_proto import api_grpc, api_pb2  # type: ignore

if TYPE_CHECKING:
    from garden_ai.client import GardenClient
else:
    GardenClient = TypeVar("GardenClient")

from ..schemas.modal import (
    ModalFunctionMetadata,
    ModalInvocationRequest,
    ModalInvocationResponse,
)


class MaximumArgumentSizeError(ValueError):
    """Raised when input arguments are too large to pass directly to modal"""

    pass


class ModalFunction:
    def __init__(
        self, metadata: ModalFunctionMetadata, client: GardenClient | None = None
    ):
        self.metadata = metadata
        self.client = client
        if self.client is None:
            from garden_ai import GardenClient

            self.client = GardenClient()

    def __call__(self, *args, **kwargs):
        # build request with serialized args
        args_kwargs_serialized = serialize((args, kwargs))
        if len(args_kwargs_serialized) > MAX_OBJECT_SIZE_BYTES:
            raise MaximumArgumentSizeError(
                "Garden's modal integration does not yet support input arguments greater than 2MiB."
            )
        request = ModalInvocationRequest(
            app_name=self.metadata.app_name,
            function_name=self.metadata.function_name,
            args_kwargs_serialized=args_kwargs_serialized,
        )
        response: ModalInvocationResponse = (
            self.client.backend_client.invoke_modal_function(request)
        )
        result_data: dict = response.result.model_dump(
            mode="python", exclude_defaults=True
        )
        modal_result_struct = api_pb2.GenericResult(**result_data)

        # HACK: need a client to deserialize, but can't assume the user has a modal account/creds
        # (also can't import from modal.config to get the url - let's hope they don't rebrand)
        with modal.Client.anonymous("https://api.modal.com") as modal_client:
            return _modal_process_result_sync(
                modal_result_struct, response.data_format, modal_client.stub
            )

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.metadata == other.metadata


def _modal_process_result_sync(
    modal_result_struct: api_pb2.GenericResult,
    modal_data_format: int,
    modal_client_stub: api_grpc.ModalClientStub,
) -> Any:
    """Helper: invoke modal's result processing/deserialization code synchronously in its own event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(
        modal._utils.function_utils._process_result(
            modal_result_struct, modal_data_format, modal_client_stub
        )
    )
