"""Schemas for Groundhog (HPC) endpoints and functions."""

from uuid import UUID

from pydantic import BaseModel, Field

from .entrypoint import (
    DatasetMetadata,
    ModelMetadata,
    NotebookMetadata,
    PaperMetadata,
    RepositoryMetadata,
)
from .schema_utils import UniqueList

# ============================================================================
# HPC Endpoint Schemas
# ============================================================================


class HpcEndpointCreateRequest(BaseModel):
    """Request schema for creating an HPC endpoint."""

    name: str
    gcmu_id: str | None = None  # Globus Compute endpoint UUID


class HpcEndpointResponse(BaseModel):
    """Response schema for an HPC endpoint."""

    id: int
    name: str
    gcmu_id: str | None = None
    owner: str | None = None
    owner_identity_id: UUID | None = None


class HpcEndpointPatchRequest(BaseModel):
    """Request schema for patching an HPC endpoint."""

    name: str | None = None
    gcmu_id: str | None = None


# ============================================================================
# HPC Function Schemas
# ============================================================================


class HpcEndpointInfo(BaseModel):
    """Basic endpoint info returned with HPC functions."""

    name: str
    gcmu_id: str | None = None


class HpcFunctionCreateRequest(BaseModel):
    """Request schema for creating an HPC function."""

    function_name: str
    endpoint_ids: list[int]
    function_text: str

    # Metadata
    title: str
    description: str | None = None
    year: str
    authors: UniqueList[str] = Field(default_factory=list)
    tags: UniqueList[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    test_functions: list[str] = Field(default_factory=list)

    # Related metadata
    models: list[ModelMetadata] = Field(default_factory=list)
    repositories: list[RepositoryMetadata] = Field(default_factory=list)
    papers: list[PaperMetadata] = Field(default_factory=list)
    datasets: list[DatasetMetadata] = Field(default_factory=list)
    notebooks: list[NotebookMetadata] = Field(default_factory=list)


class HpcFunctionResponse(BaseModel):
    """Response schema for an HPC function."""

    id: int
    function_name: str
    function_text: str | None = None

    # Metadata
    title: str | None = None
    description: str | None = None
    year: str | None = None
    authors: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    is_archived: bool = False

    # Endpoint info
    available_endpoints: list[HpcEndpointInfo] = Field(default_factory=list)
    num_invocations: int = 0

    # Related metadata
    models: list[ModelMetadata] = Field(default_factory=list)
    repositories: list[RepositoryMetadata] = Field(default_factory=list)
    papers: list[PaperMetadata] = Field(default_factory=list)
    datasets: list[DatasetMetadata] = Field(default_factory=list)
    notebooks: list[NotebookMetadata] = Field(default_factory=list)


class HpcFunctionPatchRequest(BaseModel):
    """Request schema for patching an HPC function."""

    function_name: str | None = None
    endpoint_ids: list[int] | None = None
    is_archived: bool | None = None

    title: str | None = None
    description: str | None = None
    year: str | None = None
    function_text: str | None = None

    authors: UniqueList[str] | None = None
    tags: UniqueList[str] | None = None
    requirements: list[str] | None = None
    test_functions: list[str] | None = None

    models: list[ModelMetadata] | None = None
    repositories: list[RepositoryMetadata] | None = None
    papers: list[PaperMetadata] | None = None
    datasets: list[DatasetMetadata] | None = None
    notebooks: list[NotebookMetadata] | None = None
