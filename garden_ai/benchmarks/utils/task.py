import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


class BaseBenchmarkTask:
    """
    Base class for benchmark tasks.

    Provides common utilities for:
    - Extracting model metadata (package, factory, kwargs)
    - Running benchmarks locally for testing
    """

    def __init__(
        self, adapter, repo_url: str, repo_ref: str, model_package: Optional[str] = None
    ):
        self.adapter = adapter
        self.repo_url = repo_url
        self.repo_ref = repo_ref
        self.model_package = model_package

    def _extract_model_config(
        self,
        model: Any = None,
        model_package: Optional[str] = None,
        model_factory: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Helper to resolve model configuration from either a local instance or explicit arguments.
        """
        model_checkpoint = None

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

        if model_kwargs is None:
            model_kwargs = {}

        return {
            "model_package": model_package,
            "model_factory": model_factory,
            "model_kwargs": model_kwargs,
            "model_checkpoint": model_checkpoint,
        }

    def _run_local_wrapper(
        self, runner_func_path: str, runner_func_name: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a benchmark runner function locally in a subprocess.

        Args:
            runner_func_path: Import path to the runner function (e.g. 'garden_ai.benchmarks.matbench_discovery.remote_runner')
            runner_func_name: Name of the runner function (e.g. 'run_matbench_is2re')
            config: Configuration dictionary to pass to the runner function.
        """
        results_file_path = (
            Path(tempfile.gettempdir()) / f"benchmark_results_{id(config)}.json"
        )

        wrapper_script = f'''
import json
from {runner_func_path} import {runner_func_name}

config = {repr(config)}
results = {runner_func_name}(**config)

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
