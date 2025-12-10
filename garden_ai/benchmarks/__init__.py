"""Garden AI benchmarking framework.

This module provides interfaces for running standardized benchmarks on
models hosted in Garden AI or developed locally.

Available benchmarks:
    - MatbenchDiscovery: Materials discovery benchmark suite
"""

from typing import Any, Dict, Optional

from garden_ai.client import GardenClient
from garden_ai.schemas.benchmark import BenchmarkResultCreateRequest

from .matbench_discovery.enums import DatasetSize, MatbenchTask
from .matbench_discovery.tasks import MatbenchDiscovery

__all__ = [
    "MatbenchDiscovery",
    "MatbenchTask",
    "DatasetSize",
    "publish_benchmark_result",
]


def publish_benchmark_result(
    result: Dict[str, Any],
    benchmark_name: Optional[str] = None,
    task_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Publish benchmark results to the Garden AI backend.

    This function takes the output from a benchmark task (e.g., MatbenchDiscovery.IS2RE.remote())
    and publishes it to the Garden backend for tracking and leaderboard purposes.

    Args:
        result: The output dictionary from a benchmark task. Should contain:
            - 'metrics': Dictionary of benchmark metrics (F1, DAF, MAE, etc.)
            - 'run_metadata': Optional run metadata (hardware, timing, cost)
            - '_benchmark_info': Auto-injected benchmark/task info (if from wrapped method)
        benchmark_name: Override for benchmark name (defaults to auto-detected from result)
        task_name: Override for task name (defaults to auto-detected from result)

    Returns:
        Dictionary containing the response from the backend, including the result ID.

    Raises:
        ValueError: If benchmark_name or task_name cannot be determined.
        requests.HTTPError: If the backend request fails.

    Example:
        ```python
        from garden_ai.benchmarks import MatbenchDiscovery, publish_benchmark_result

        # Run a benchmark
        output = MatbenchDiscovery.IS2RE.remote(
            endpoint="your-endpoint-id",
            model_factory=create_model,
            model_packages="mace-torch",
        )

        # Publish the results
        response = publish_benchmark_result(output)
        print(f"Published with ID: {response['id']}")
        ```
    """
    # Extract benchmark info from result or use provided overrides
    benchmark_info = result.get("_benchmark_info", {})

    final_benchmark_name = benchmark_name or benchmark_info.get("benchmark_name")
    final_task_name = task_name or benchmark_info.get("task_name")

    if not final_benchmark_name:
        raise ValueError(
            "benchmark_name is required. Either pass it explicitly or use a result "
            "from a MatbenchDiscovery task method (e.g., MatbenchDiscovery.IS2RE.remote())."
        )

    if not final_task_name:
        raise ValueError(
            "task_name is required. Either pass it explicitly or use a result "
            "from a MatbenchDiscovery task method (e.g., MatbenchDiscovery.IS2RE.remote())."
        )

    # Extract metrics and run_metadata
    metrics = result.get("metrics", {})
    run_metadata = result.get("run_metadata")

    if not metrics:
        raise ValueError("Result must contain 'metrics' dictionary.")

    # Create the request payload
    payload = BenchmarkResultCreateRequest(
        benchmark_name=final_benchmark_name,
        benchmark_task_name=final_task_name,
        metrics=metrics,
        run_metadata=run_metadata,
    )

    # Get authenticated client and publish
    client = GardenClient()
    response = client.backend_client.publish_benchmark_result(payload)
    return response.model_dump()
