"""Matbench Discovery benchmark task implementations."""

from typing import TYPE_CHECKING, Any

from .remote_runner import run_matbench_is2re

if TYPE_CHECKING:
    from . import MatbenchDiscovery


class IS2RETask:
    """Initial Structure to Relaxed Energy benchmark task.

    This task evaluates a model's ability to predict the relaxed energy
    and geometry of crystal structures starting from unrelaxed initial
    configurations.

    The task:
    1. Loads initial (unrelaxed) structures from the WBM test set
    2. Uses the model to perform geometry optimization
    3. Records final energies and relaxed structures
    4. Calculates metrics comparing to DFT ground truth
    """

    def __init__(
        self,
        adapter: "MatbenchDiscovery",
        repo_url: str,
        repo_ref: str,
        model_package: str | None = None,
    ):
        """Initialize IS2RE task.

        Args:
            adapter: MatbenchDiscovery adapter instance
            repo_url: Matbench Discovery repository URL
            repo_ref: Git ref (branch/tag/commit) to use
            model_package: Default model package to install (can override in submit)
        """
        self.adapter = adapter
        self.repo_url = repo_url
        self.repo_ref = repo_ref
        self.model_package = model_package
        self.name = "IS2RE"

    def submit(
        self,
        model=None,
        num_structures: int = 100,
        model_package: str | None = None,
        model_factory: str | None = None,
        model_kwargs: dict | None = None,
        use_multi_gpu: bool = True,
    ):
        """Submit IS2RE benchmark job to remote executor.

        You can specify the model in two ways:
        1. Pass a local model instance (will introspect to get remote construction info)
        2. Explicitly specify model_package and model_factory

        Args:
            model: (Optional) Local model instance. If provided, will extract
                   package, class, and checkpoint information from it.
            num_structures: Number of test structures to evaluate (default: 100).
                           Full test set has ~257k structures. Use smaller values
                           for quick testing.
            model_package: Python package name to install (e.g., "mace-torch").
                          Required if model is None.
            model_factory: How to instantiate the model on remote. Can be:
                          - Function name: "mace_mp" (will call as function)
                          - Class name: "MACE" (will instantiate as class)
                          Required if model is None.
            model_kwargs: Dictionary of kwargs to pass when creating model remotely.
                         Example: {"model": "medium", "device": "cuda"}
            use_multi_gpu: If True, automatically detect and use all available GPUs
                          in parallel for faster processing. If False, use single
                          GPU/CPU. (default: True)

        Returns:
            Future object that will contain benchmark results when complete.
            Call .result() to block and wait for completion.

        Examples:
            Using local model instance:
            >>> from mace.calculators import mace_mp
            >>> model = mace_mp(model="medium")
            >>> future = task.submit(model, num_structures=50)

            Specifying remote construction explicitly:
            >>> future = task.submit(
            ...     model_package="mace-torch",
            ...     model_factory="mace_mp",
            ...     model_kwargs={"model": "medium", "device": "cuda"},
            ...     num_structures=50,
            ...     use_multi_gpu=True
            ... )
        """
        # Determine how to construct model remotely
        if model is not None:
            # Extract info from local model instance
            if model_package is None:
                if self.model_package is not None:
                    model_package = self.model_package
                else:
                    # Infer from model's module
                    model_package = model.__class__.__module__.split(".")[0]

            if model_factory is None:
                model_factory = model.__class__.__name__

            # Get checkpoint path if model has one
            model_checkpoint = None
            if hasattr(model, "checkpoint_path"):
                model_checkpoint = model.checkpoint_path
            elif hasattr(model, "checkpoint"):
                model_checkpoint = model.checkpoint

            # Try to extract initialization kwargs if available
            if model_kwargs is None and hasattr(model, "_init_kwargs"):
                model_kwargs = model._init_kwargs

        else:
            # Must provide explicit construction info
            if model_package is None or model_factory is None:
                raise ValueError(
                    "If model is not provided, must specify both "
                    "model_package and model_factory"
                )
            model_checkpoint = None

        if model_kwargs is None:
            model_kwargs = {}

        # Get executor (will create if needed) and submit remote execution
        executor = self.adapter._get_executor()
        future = executor.submit(
            run_matbench_is2re,
            repo_url=self.repo_url,
            repo_ref=self.repo_ref,
            model_package=model_package,
            model_factory=model_factory,
            model_kwargs=model_kwargs,
            model_checkpoint=model_checkpoint,
            num_structures=num_structures,
            use_multi_gpu=use_multi_gpu,
        )

        return future

    def local(
        self,
        model=None,
        num_structures: int = 100,
        model_package: str | None = None,
        model_factory: str | None = None,
        model_kwargs: dict | None = None,
        use_multi_gpu: bool = True,
    ) -> dict:
        """Run benchmark locally in ephemeral UV environment.

        This executes the same benchmark workflow locally instead of submitting
        to a remote Globus Compute endpoint. Useful for testing and development.

        Args:
            model: Optional local model instance to extract metadata from
            num_structures: Number of test structures to evaluate
            model_package: Python package name to install (e.g., "mace-torch")
            model_factory: Function or class name to create model
            model_kwargs: Dictionary of kwargs for model creation
            use_multi_gpu: If True, automatically detect and use all available GPUs
                          in parallel. If False, use single GPU/CPU. (default: True)

        Returns:
            Dictionary with benchmark results (same format as remote execution)

        Example:
            >>> results = task.local(
            ...     model_package="mace-torch",
            ...     model_factory="mace_mp",
            ...     model_kwargs={"model": "medium", "device": "cpu"},
            ...     num_structures=10,
            ...     use_multi_gpu=False
            ... )
        """
        import json
        import subprocess
        import tempfile
        from pathlib import Path

        # Extract model metadata if model instance provided
        if model is not None:
            if model_package is None:
                if self.model_package is not None:
                    model_package = self.model_package
                else:
                    model_package = model.__class__.__module__.split(".")[0]

            if model_factory is None:
                model_factory = model.__class__.__name__

            model_checkpoint = None
            if hasattr(model, "checkpoint_path"):
                model_checkpoint = model.checkpoint_path
            elif hasattr(model, "checkpoint"):
                model_checkpoint = model.checkpoint

            if model_kwargs is None and hasattr(model, "_init_kwargs"):
                model_kwargs = model._init_kwargs
        else:
            if model_package is None or model_factory is None:
                raise ValueError(
                    "If model is not provided, must specify both "
                    "model_package and model_factory"
                )
            model_checkpoint = None

        if model_kwargs is None:
            model_kwargs = {}

        # Run benchmark in subprocess with isolated environment
        import sys

        config = {
            "repo_url": self.repo_url,
            "repo_ref": self.repo_ref,
            "model_package": model_package,
            "model_factory": model_factory,
            "model_kwargs": model_kwargs,
            "model_checkpoint": model_checkpoint,
            "num_structures": num_structures,
            "use_multi_gpu": use_multi_gpu,
        }

        results_file_path = (
            Path(tempfile.gettempdir()) / f"benchmark_results_{id(config)}.json"
        )

        wrapper_script = f'''
import json
from garden_ai.benchmarks.matbench_discovery.remote_runner import run_matbench_is2re

config = {repr(config)}
results = run_matbench_is2re(**config)

with open("{results_file_path}", "w") as f:
    json.dump(results, f, indent=2)
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapper_script)
            wrapper_path = f.name

        try:
            # Run without capturing output so logs stream to console in real-time
            result = subprocess.run(
                [sys.executable, wrapper_path],
                timeout=3600,
                # Don't capture output - let it stream to console
                stdout=None,
                stderr=None,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Local benchmark failed with return code {result.returncode}"
                )

            if not results_file_path.exists():
                raise RuntimeError(
                    f"Benchmark results file not found at {results_file_path}"
                )

            with open(results_file_path) as f:
                return json.load(f)

        finally:
            Path(wrapper_path).unlink(missing_ok=True)
            results_file_path.unlink(missing_ok=True)

    def calculate_metrics(self, outputs: dict) -> dict[str, Any]:
        # TODO: implement the full metrics calculation,
        # this is just a placeholder for now
        """Calculate benchmark metrics from raw outputs.

        For MVP, this returns basic statistics. Future versions will compare
        against DFT ground truth and calculate proper benchmark metrics like
        F1 score, discovery yield, etc.

        Args:
            outputs: Dictionary from remote execution containing:
                - energies: List of relaxed energies
                - num_converged: Number of successful relaxations
                - failed_indices: Indices of failed structures

        Returns:
            Dictionary of calculated metrics:
                - num_attempted: Total structures attempted
                - num_converged: Number of successful relaxations
                - success_rate: Fraction of successful relaxations
                - mean_energy: Average final energy (eV/atom, if available)
                - num_failed: Count of failed relaxations
        """
        energies = outputs.get("energies", [])
        num_converged = outputs.get("num_converged", 0)
        failed_indices = outputs.get("failed_indices", [])

        # Filter out None values (failed relaxations)
        valid_energies = [e for e in energies if e is not None]

        metrics = {
            "num_attempted": len(energies),
            "num_converged": num_converged,
            "num_failed": len(failed_indices),
            "success_rate": num_converged / len(energies) if energies else 0.0,
        }

        # Calculate energy statistics if we have valid results
        if valid_energies:
            metrics["mean_energy"] = sum(valid_energies) / len(valid_energies)
            metrics["min_energy"] = min(valid_energies)
            metrics["max_energy"] = max(valid_energies)

        return metrics
