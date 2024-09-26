from .schema_utils import B64Bytes
from pydantic import BaseModel


class ModalFunctionMetadata(BaseModel):
    app_name: str
    function_name: str
    # TODO other fields we'll want to persist


class _ModalGenericResult(BaseModel):
    # duplicates key fields from modal's protobuf api_pb2.GenericResult type, so our sdk can
    # build one manually and leave the rest of the result processing to modal
    status: int
    exception: str = ""
    traceback: str = ""
    serialized_tb: B64Bytes = b""
    tb_line_cache: B64Bytes = b""
    data: B64Bytes = b""
    data_blob_id: str = ""


class ModalInvocationRequest(BaseModel):
    app_name: str
    function_name: str
    args_kwargs_serialized: B64Bytes


class ModalInvocationResponse(BaseModel):
    result: _ModalGenericResult
    data_format: int
