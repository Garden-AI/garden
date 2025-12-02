"""Matbench Discovery benchmark adapter for Garden AI.

This module provides a clean interface for running Matbench Discovery benchmarks
on remote HPC systems via Globus Compute. It handles environment setup,
dependency installation, and benchmark execution.

Example usage:
    >>> from garden_ai.benchmarks.matbench_discovery import MatbenchDiscovery
    >>> from my_model import MyModel
    >>>
    >>> # Configure for your HPC endpoint
    >>> endpoint_id = "your-endpoint-uuid"
    >>> endpoint_config = {
    ...     "account": "project-account",
    ...     "partition": "gpu",
    ...     "scheduler_options": "#SBATCH --gpus-per-node=1"
    ... }
    >>>
    >>> # Run benchmark
    >>> with MatbenchDiscovery(endpoint_id, endpoint_config) as bench:
    ...     model = MyModel()
    ...     task = bench.tasks.IS2RE
    ...     future = task.submit(model, num_structures=100)
    ...     results = future.result()
    ...     metrics = task.calculate_metrics(results)
    ...     print(metrics)
"""

from typing import Any

from globus_compute_sdk import Executor
from globus_compute_sdk.serialize import CombinedCode, ComputeSerializer

from .enums import DatasetSize, MatbenchTask
from .tasks import (
    IP2ETask,
    IS2ETask,
    IS2RETask,
    RP2RETask,
    RS2RETask,
    S2EFSMTask,
    S2EFSTask,
    S2EFTask,
    S2ETask,
    S2RETask,
)

__all__ = [
    "MatbenchDiscovery",
    "MatbenchTask",
    "DatasetSize",
    "IS2RETask",
    "RS2RETask",
    "S2EFSTask",
    "S2EFTask",
    "S2EFSMTask",
    "IS2ETask",
    "S2ETask",
    "S2RETask",
    "RP2RETask",
    "IP2ETask",
]


class MatbenchDiscovery:
    """Adapter for running Matbench Discovery benchmarks locally or remotely.

    This class manages the lifecycle of benchmark execution:
    - Provides access to benchmark tasks (IS2RE, etc.)
    - For remote execution: creates and manages Globus Compute executor
    - For local execution: runs in ephemeral UV environment

    Use as a context manager to ensure proper cleanup:
        # Local execution
        with MatbenchDiscovery() as bench:
            result = bench.tasks.IS2RE.local(...)

        # Remote execution
        with MatbenchDiscovery(endpoint_id="uuid", endpoint_config={...}) as bench:
            future = bench.tasks.IS2RE.submit(...)

    Attributes:
        tasks: Namespace containing available benchmark tasks
            - tasks.IS2RE: Initial Structure to Relaxed Energy task
    """

    # Matbench Discovery repository configuration
    REPO_URL = "https://github.com/janosh/matbench-discovery"
    REPO_REF = "main"
    PYTHON_VERSION = "3.11"

    def __init__(
        self,
        endpoint_id: str | None = None,
        user_endpoint_config: dict[str, Any] | None = None,
        repo_ref: str | None = None,
        model_package: str | None = None,
    ):
        """Initialize Matbench Discovery adapter.

        Args:
            endpoint_id: Globus Compute endpoint UUID for remote execution.
                        If None, only local execution (.local()) is available.
            user_endpoint_config: Optional HPC configuration for remote endpoint.
                                 Example for SLURM:
                                 {
                                     "account": "project-account",
                                     "partition": "gpu-debug",
                                     "scheduler_options": "#SBATCH --gpus-per-node=1"
                                 }
            repo_ref: Git branch/tag/commit to use (default: "main")
            model_package: Default model package to install for all tasks
                          (can be overridden per task)
        """
        self.endpoint_id = endpoint_id
        self.user_endpoint_config = user_endpoint_config or {}

        # Ensure 'requirements' is present to avoid endpoint template errors
        if "requirements" not in self.user_endpoint_config:
            self.user_endpoint_config["requirements"] = ""

        self.repo_ref = repo_ref or self.REPO_REF
        self.model_package = model_package

        # Executor is created lazily on first submit() call
        self._executor: Executor | None = None
        self.tasks: Any = None

    def _get_executor(self) -> Executor:
        """Get or create the Globus Compute executor (lazy initialization).

        Returns:
            Executor instance

        Raises:
            ValueError: If endpoint_id was not provided during initialization
        """
        if self._executor is None:
            if self.endpoint_id is None:
                raise ValueError(
                    "endpoint_id is required for remote execution. "
                    "Either provide endpoint_id during initialization or use .local() method."
                )

            executor_kwargs = {"endpoint_id": self.endpoint_id}
            if self.user_endpoint_config:
                executor_kwargs["user_endpoint_config"] = self.user_endpoint_config

            # Use CombinedCode serialization to send actual function code
            # rather than module references (avoids needing garden_ai installed remotely)
            executor_kwargs["serializer"] = ComputeSerializer(
                strategy_code=CombinedCode()
            )

            self._executor = Executor(**executor_kwargs)

        return self._executor

    def __enter__(self):
        """Set up tasks when entering context."""
        # Initialize tasks - executor will be created lazily when needed
        # Using a simple namespace object for dot access
        self.tasks = type(
            "Tasks",
            (),
            {
                "IS2RE": IS2RETask(
                    adapter=self,
                    repo_url=self.REPO_URL,
                    repo_ref=self.repo_ref,
                    model_package=self.model_package,
                ),
                "RS2RE": RS2RETask(
                    adapter=self,
                    repo_url=self.REPO_URL,
                    repo_ref=self.repo_ref,
                    model_package=self.model_package,
                ),
                "S2EFS": S2EFSTask(
                    adapter=self,
                    repo_url=self.REPO_URL,
                    repo_ref=self.repo_ref,
                    model_package=self.model_package,
                ),
                "S2EF": S2EFTask(
                    adapter=self,
                    repo_url=self.REPO_URL,
                    repo_ref=self.repo_ref,
                    model_package=self.model_package,
                ),
                "S2EFSM": S2EFSMTask(
                    adapter=self,
                    repo_url=self.REPO_URL,
                    repo_ref=self.repo_ref,
                    model_package=self.model_package,
                ),
                "IS2E": IS2ETask(
                    adapter=self,
                    repo_url=self.REPO_URL,
                    repo_ref=self.repo_ref,
                    model_package=self.model_package,
                ),
                "S2E": S2ETask(
                    adapter=self,
                    repo_url=self.REPO_URL,
                    repo_ref=self.repo_ref,
                    model_package=self.model_package,
                ),
                "S2RE": S2RETask(
                    adapter=self,
                    repo_url=self.REPO_URL,
                    repo_ref=self.repo_ref,
                    model_package=self.model_package,
                ),
                "RP2RE": RP2RETask(
                    adapter=self,
                    repo_url=self.REPO_URL,
                    repo_ref=self.repo_ref,
                    model_package=self.model_package,
                ),
                "IP2E": IP2ETask(
                    adapter=self,
                    repo_url=self.REPO_URL,
                    repo_ref=self.repo_ref,
                    model_package=self.model_package,
                ),
            },
        )()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up executor when exiting context."""
        if self._executor:
            self._executor.shutdown(wait=True)
        return False  # Don't suppress exceptions
