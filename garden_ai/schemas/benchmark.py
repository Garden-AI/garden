"""Benchmark-related schemas for API requests/responses."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class BenchmarkResultCreateRequest(BaseModel):
    """Request schema for publishing benchmark results to the backend."""

    benchmark_name: str = Field(
        ...,
        description="Name of the benchmark suite (e.g., 'matbench_discovery')",
    )
    benchmark_task_name: str = Field(
        ...,
        description="Name of the specific task within the benchmark (e.g., 'IS2RE', 'S2EFS')",
    )
    metrics: Dict[str, Any] = Field(
        ...,
        description="Dictionary of benchmark metrics (F1, DAF, MAE, etc.)",
    )
    run_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional run metadata (hardware info, timing, cost estimates)",
    )


class BenchmarkResultResponse(BaseModel):
    """Response schema from the benchmark result creation endpoint."""

    id: int = Field(..., description="Unique identifier for the benchmark result")
    benchmark_name: str
    benchmark_task_name: str
    metrics: Dict[str, Any]
    run_metadata: Optional[Dict[str, Any]] = None
