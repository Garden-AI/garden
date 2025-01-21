from enum import Enum
from pydantic import BaseModel, Field, model_validator

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
    data: B64Bytes | None = None

    data_blob_url: str | None = None

    @model_validator(mode="after")
    def one_of_data_or_blob_url(self):
        assert not (
            self.data and self.data_blob_url
        ), "Only one of data or data_blob_url should be set."
        return self


class ModalInvocationRequest(BaseModel):
    function_id: int
    args_kwargs_serialized: B64Bytes | None = None
    args_blob_id: str | None = None

    @model_validator(mode="after")
    def one_of_args_or_blob(self):
        assert not (
            self.args_kwargs_serialized and self.args_blob_id
        ), "Only one of args_kwargs_serialized or args_blob_id should be set."
        return self


class ModalInvocationResponse(BaseModel):
    result: _ModalGenericResult
    data_format: int


class AsyncModalInvocationResponse(BaseModel):
    id: int
    status: str


class ModalBlobUploadURLRequest(BaseModel):
    content_length: int
    content_md5: str
    content_sha256_base64: str


class _UploadType(str, Enum):
    SINGLE = "single"
    MULTIPART = "multipart"


class _MultiPartUpload(BaseModel):
    part_length: int
    upload_urls: list[str]
    completion_url: str


class ModalBlobUploadURLResponse(BaseModel):
    # imitating the modal BlobCreate response payload
    blob_id: str
    upload_type: _UploadType

    upload_url: str | None = None
    multipart: _MultiPartUpload | None = None
