"""Schemas for HPC functions and related metadata."""

from pydantic import BaseModel, Field


class HpcFunctionMetadata(BaseModel):
    """Metadata for an HPC function."""

    id: int
    function_name: str
    function_text: str
    title: str | None = None
    description: str | None = None
    available_endpoints: list[dict[str, str]] = Field(default_factory=list)
    num_invocations: int = 0

    # Additional metadata from CommonFunctionMetadata
    is_archived: bool = False
    year: str | None = None
    authors: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class HpcInvocationCreateRequest(BaseModel):
    """Request schema for creating an HPC invocation log."""

    function_id: int
    endpoint_gcmu_id: str
    globus_task_id: str
    user_endpoint_config: dict = Field(default_factory=dict)
