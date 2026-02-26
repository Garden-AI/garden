"""Schemas for Groundhog (HPC) endpoints and functions."""

from uuid import UUID

from pydantic import BaseModel, Field

from .base import (
    BaseFunctionCreateRequest,
    BaseFunctionPatchRequest,
    HpcEndpointBase,
    RelatedMetadata,
)


class HpcEndpointCreateRequest(HpcEndpointBase):
    """Request schema for creating an HPC endpoint."""

    pass


class HpcEndpointResponse(HpcEndpointBase):
    """Response schema for an HPC endpoint."""

    id: int
    owner: str | None = None
    owner_identity_id: UUID | None = None


class HpcEndpointPatchRequest(BaseModel):
    """Request schema for patching an HPC endpoint."""

    name: str | None = None
    gcmu_id: str | None = None


class HpcEndpointInfo(HpcEndpointBase):
    """Basic endpoint info returned with HPC functions."""

    pass


class HpcFunctionCreateRequest(BaseFunctionCreateRequest):
    """Request schema for creating an HPC function."""

    endpoint_ids: list[int]


class HpcFunctionResponse(RelatedMetadata):
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


class HpcFunctionPatchRequest(BaseFunctionPatchRequest):
    """Request schema for patching an HPC function."""

    endpoint_ids: list[int] | None = None
    is_archived: bool | None = None
