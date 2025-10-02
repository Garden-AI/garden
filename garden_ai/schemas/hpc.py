"""Schemas for HPC functions and related metadata."""

from pydantic import BaseModel, Field


class HpcDeploymentInfo(BaseModel):
    """Information about HPC deployment for a function on a specific endpoint."""

    deployment_id: int
    endpoint_name: str
    endpoint_gcmu_id: str
    conda_env_path: str


class HpcFunctionMetadata(BaseModel):
    """Metadata for an HPC function."""

    id: int
    function_name: str
    function_text: str
    title: str | None = None
    description: str | None = None
    available_deployments: list[HpcDeploymentInfo] = Field(default_factory=list)
    available_endpoints: list[str] = Field(default_factory=list)
    num_invocations: int = 0

    # Additional metadata from CommonFunctionMetadata
    is_archived: bool = False
    year: str | None = None
    authors: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
