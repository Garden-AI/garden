"""Garden AI benchmarking framework.

This module provides interfaces for running standardized benchmarks on
models hosted in Garden AI or developed locally.

Available benchmarks:
    - MatbenchDiscovery: Materials discovery benchmark suite
"""

from .matbench_discovery import IS2RETask, MatbenchDiscovery, MatbenchTask

__all__ = [
    "MatbenchDiscovery",
    "MatbenchTask",
    "IS2RETask",
]


def publish_benchmark_result(benchmark, model, results):
    """Publish benchmark results to Garden AI backend.

    This is a placeholder for future functionality to store benchmark
    results alongside published models.

    Args:
        benchmark: Benchmark adapter instance
        model: Model that was benchmarked
        results: Dictionary of benchmark metrics
    """
    # TODO: Implement when backend API is ready
    raise NotImplementedError(
        "Publishing benchmark results is not yet implemented. "
        "For now, save results locally or to your own storage."
    )
