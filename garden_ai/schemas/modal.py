from pydantic import BaseModel, Field

from .entrypoint import (
    DatasetMetadata,
    ModelMetadata,
    PaperMetadata,
    RepositoryMetadata,
)
from .schema_utils import B64Bytes, UniqueList


class ModalFunctionMetadata(BaseModel):
    # Identifiers
    id: int
    doi: str | None = None

    # DataCite Metadata
    title: str
    description: str | None = ""
    year: str

    # Function Metadata
    is_archived: bool = False
    function_name: str
    function_text: str

    authors: UniqueList[str] = Field(default_factory=list)
    tags: UniqueList[str] = Field(default_factory=list)
    test_functions: list[str] = Field(default_factory=list)

    models: list[ModelMetadata] = Field(default_factory=list)
    repositories: list[RepositoryMetadata] = Field(default_factory=list)
    papers: list[PaperMetadata] = Field(default_factory=list)
    datasets: list[DatasetMetadata] = Field(default_factory=list)


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
    function_id: int
    args_kwargs_serialized: B64Bytes


class ModalInvocationResponse(BaseModel):
    result: _ModalGenericResult
    data_format: int


class AsyncModalInvocationResponse(BaseModel):
    id: int
    status: str
