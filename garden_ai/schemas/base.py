"""Base schemas for common function and endpoint metadata."""

from pydantic import BaseModel, Field

from .entrypoint import (
    DatasetMetadata,
    ModelMetadata,
    NotebookMetadata,
    PaperMetadata,
    RepositoryMetadata,
)
from .schema_utils import UniqueList


class RelatedMetadata(BaseModel):
    """Related resources (models, papers, datasets, etc.) for a function."""

    models: list[ModelMetadata] = Field(default_factory=list)
    repositories: list[RepositoryMetadata] = Field(default_factory=list)
    papers: list[PaperMetadata] = Field(default_factory=list)
    datasets: list[DatasetMetadata] = Field(default_factory=list)
    notebooks: list[NotebookMetadata] = Field(default_factory=list)


class BaseFunctionMetadata(BaseModel):
    """Core metadata fields shared by all function types (Modal and HPC)."""

    function_name: str
    title: str
    description: str | None = None
    year: str
    authors: UniqueList[str] = Field(default_factory=list)
    tags: UniqueList[str] = Field(default_factory=list)
    function_text: str
    requirements: list[str] = Field(default_factory=list)
    test_functions: list[str] = Field(default_factory=list)


class BaseFunctionCreateRequest(BaseFunctionMetadata, RelatedMetadata):
    """Base for function creation requests with metadata and related resources."""

    pass


class BaseFunctionPatchRequest(BaseModel):
    """Base for function patch requests - all fields optional."""

    function_name: str | None = None
    title: str | None = None
    description: str | None = None
    year: str | None = None
    function_text: str | None = None
    authors: UniqueList[str] | None = None
    tags: UniqueList[str] | None = None
    requirements: list[str] | None = None
    test_functions: list[str] | None = None

    # Related metadata
    models: list[ModelMetadata] | None = None
    repositories: list[RepositoryMetadata] | None = None
    papers: list[PaperMetadata] | None = None
    datasets: list[DatasetMetadata] | None = None
    notebooks: list[NotebookMetadata] | None = None


class HpcEndpointBase(BaseModel):
    """Base fields for HPC endpoint schemas."""

    name: str
    gcmu_id: str | None = None  # Globus Compute endpoint UUID
