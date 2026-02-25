"""Schemas for Modal App CRUD operations."""

from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field, computed_field

from .entrypoint import (
    DatasetMetadata,
    ModelMetadata,
    NotebookMetadata,
    PaperMetadata,
    RepositoryMetadata,
)
from .schema_utils import UniqueList


class ModalFunctionCreateMetadata(BaseModel):
    """Metadata for a Modal function when creating a Modal App."""

    function_name: str
    title: str
    description: str | None = None
    year: str
    authors: UniqueList[str] = Field(default_factory=list)
    tags: UniqueList[str] = Field(default_factory=list)
    function_text: str
    example_usage: str = ""
    requirements: list[str] = Field(default_factory=list)
    test_functions: list[str] = Field(default_factory=list)

    # Related metadata
    models: list[ModelMetadata] = Field(default_factory=list)
    repositories: list[RepositoryMetadata] = Field(default_factory=list)
    papers: list[PaperMetadata] = Field(default_factory=list)
    datasets: list[DatasetMetadata] = Field(default_factory=list)
    notebooks: list[NotebookMetadata] = Field(default_factory=list)


class ModalAppCreateRequest(BaseModel):
    """Request schema for creating a Modal App."""

    app_name: str
    file_contents: str
    base_image_name: str
    requirements: list[str] = Field(default_factory=list)
    conda_requirements: list[str] = Field(default_factory=list)
    modal_functions: list[ModalFunctionCreateMetadata] = Field(default_factory=list)
    owner_identity_id: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def modal_function_names(self) -> list[str]:
        return [mf.function_name for mf in self.modal_functions]


class AsyncModalJobStatus(str, Enum):
    """Status of an async Modal deployment job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    DONE = "done"


class ModalFunctionResponse(BaseModel):
    """Response schema for a Modal function."""

    id: int
    modal_app_id: int
    function_name: str
    doi: str | None = None

    title: str
    description: str | None = None
    year: str
    authors: UniqueList[str] = Field(default_factory=list)
    tags: UniqueList[str] = Field(default_factory=list)
    function_text: str | None = None
    example_usage: str = ""
    requirements: list[str] = Field(default_factory=list)
    is_archived: bool = False

    hardware_spec: dict = Field(default_factory=dict)
    num_invocations: int = 0

    owner: str = ""
    owner_identity_id: UUID | None = None

    # Related metadata
    models: list[ModelMetadata] = Field(default_factory=list)
    repositories: list[RepositoryMetadata] = Field(default_factory=list)
    papers: list[PaperMetadata] = Field(default_factory=list)
    datasets: list[DatasetMetadata] = Field(default_factory=list)
    notebooks: list[NotebookMetadata] = Field(default_factory=list)


class ModalAppResponse(BaseModel):
    """Response schema for a Modal App."""

    id: int
    app_name: str
    original_app_name: str | None = None
    base_image_name: str
    requirements: list[str] = Field(default_factory=list)
    conda_requirements: list[str] = Field(default_factory=list)
    file_contents: str | None = None
    modal_functions: list[ModalFunctionResponse] = Field(default_factory=list)
    owner_identity_id: UUID

    deploy_status: AsyncModalJobStatus | None = None
    deploy_error: str | None = None
    suggested_fix: str | None = None
    deployment_output: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def modal_function_ids(self) -> list[int]:
        return [mf.id for mf in self.modal_functions]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def modal_function_names(self) -> list[str]:
        return [mf.function_name for mf in self.modal_functions]


class ModalAppPatchRequest(BaseModel):
    """Request schema for patching a Modal App."""

    base_image_name: str | None = None
    requirements: list[str] | None = None
    conda_requirements: list[str] | None = None
    file_contents: str | None = None


class ModalFunctionPatchRequest(BaseModel):
    """Request schema for patching a Modal function."""

    doi: str | None = None
    function_name: str | None = None
    is_archived: bool | None = None

    title: str | None = None
    description: str | None = None
    year: str | None = None
    function_text: str | None = None
    example_usage: str | None = None

    authors: UniqueList[str] | None = None
    tags: UniqueList[str] | None = None
    test_functions: list[str] | None = None
    requirements: list[str] | None = None

    models: list[ModelMetadata] | None = None
    repositories: list[RepositoryMetadata] | None = None
    papers: list[PaperMetadata] | None = None
    datasets: list[DatasetMetadata] | None = None
    notebooks: list[NotebookMetadata] | None = None
