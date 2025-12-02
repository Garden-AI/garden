"""Matbench Discovery benchmark task implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from ..utils.remote_execution import run_remote_benchmark
from ..utils.script_builder import BenchmarkScriptBuilder
from ..utils.task import BaseBenchmarkTask

if TYPE_CHECKING:
    from . import MatbenchDiscovery
    from .enums import DatasetConfig, DatasetSize

from .metrics import classify_stable, stable_metrics

# ------------------------------------------------------------------------------
# REMOTE FUNCTIONS
# These functions are injected into the remote script.
# They must be self-contained (imports inside or provided by builder).
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# REMOTE FUNCTIONS
# These functions are injected into the remote script.
# They are now imported from remote.py to keep this file clean.
# ------------------------------------------------------------------------------
from .remote import (
    _load_dataset_common,
    _process_batch_common,
    calculate_metrics_energy,
    calculate_metrics_forces,
    get_material_ids_for_subset,
    load_dataset_mp_trj,
    load_dataset_wbm_initial,
    load_dataset_wbm_relaxed,
    load_model,
    process_batch_forces,
    process_batch_relaxation,
    process_batch_static,
)

# ------------------------------------------------------------------------------
# Task Classes
# ------------------------------------------------------------------------------


class MatbenchTask(BaseBenchmarkTask):
    """Base class for Matbench Discovery tasks."""

    def __init__(
        self,
        adapter: "MatbenchDiscovery",
        repo_url: str,
        repo_ref: str,
        model_package: str | None = None,
        task_name: str = "unknown",
    ):
        super().__init__(adapter, repo_url, repo_ref, model_package)
        self.name = task_name

    def calculate_metrics(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve metrics from the remote output."""
        return output.get("metrics", {})

    def _build_script(
        self, process_fn, load_dataset_fn, calc_metrics_fn, model_factory
    ) -> str:
        """Build the remote execution script with specific functions.

        Args:
            process_fn: Task-specific process_batch function
            load_dataset_fn: Task-specific load_dataset function
            calc_metrics_fn: Task-specific calculate_metrics function
            model_factory: User-provided function that creates the model
        """
        builder = BenchmarkScriptBuilder()

        # Add global model cache
        builder.add_preamble("_MODEL_CACHE = None")

        # Common imports
        builder.add_import(
            "from typing import List, Dict, Any, Tuple, Optional, Callable"
        )
        builder.add_import("import torch")
        builder.add_import("from ase.optimize import FIRE")
        builder.add_import("from ase.io import read")
        builder.add_import("from matbench_discovery.data import DataFiles")
        builder.add_import("from zipfile import ZipFile")
        builder.add_import("from io import TextIOWrapper")
        builder.add_import("import pandas as pd")
        builder.add_import("import numpy as np")
        builder.add_import("from collections.abc import Sequence")
        builder.add_import("from sklearn.metrics import r2_score")

        # Add user's model factory (renamed to load_model_user so load_model can call it)
        builder.add_function(model_factory, name="load_model_user")

        # Add our load_model wrapper that calls load_model_user
        builder.add_function(load_model)

        # Add helper function for dataset subset filtering
        builder.add_function(get_material_ids_for_subset)

        # Add common helpers
        builder.add_function(_process_batch_common)
        builder.add_function(_load_dataset_common)

        # Add task-specific functions with standard names expected by runner
        builder.add_function(process_fn, name="process_batch")
        builder.add_function(load_dataset_fn, name="load_dataset")
        builder.add_function(calc_metrics_fn, name="calculate_metrics_remote")

        # Inject metrics helper functions
        builder.add_function(classify_stable)
        builder.add_function(stable_metrics)

        return builder.build()

    def _prepare_runner_config(
        self, num_structures: int | "DatasetSize" | "DatasetConfig"
    ) -> Dict[str, Any]:
        """Prepare the runner configuration based on num_structures."""
        from .enums import DatasetConfig, DatasetSize

        if isinstance(num_structures, DatasetSize):
            return {
                "repo_url": self.repo_url,
                "repo_ref": self.repo_ref,
                "dataset_subset": num_structures.value,
                "dataset_seed": 42,
            }
        elif isinstance(num_structures, DatasetConfig):
            return {
                "repo_url": self.repo_url,
                "repo_ref": self.repo_ref,
                "dataset_subset": num_structures.subset.value,
                "dataset_seed": num_structures.seed,
            }
        else:
            return {
                "repo_url": self.repo_url,
                "repo_ref": self.repo_ref,
                "num_structures": num_structures,
                "dataset_subset": "full",
            }

    def _prepare_dependencies(self, model_packages: str | List[str]) -> List[str]:
        """Prepare the list of dependencies."""
        packages = (
            [model_packages] if isinstance(model_packages, str) else model_packages
        )
        return ["matbench-discovery>=1.3.0"] + packages

    def _generate_checkpoint_name(
        self, model_packages: str | List[str], runner_config: Dict[str, Any]
    ) -> str:
        """Generate a unique checkpoint name."""
        import time
        import uuid

        model_str = (
            str(model_packages)
            .replace("[", "")
            .replace("]", "")
            .replace("'", "")
            .replace('"', "")
            .replace(",", "_")
            .replace(" ", "")
        )
        subset_str = runner_config.get("dataset_subset", "custom")
        timestamp = int(time.time())
        short_uuid = str(uuid.uuid4())[:8]
        return f"matbench_{model_str}_{subset_str}_{timestamp}_{short_uuid}.json"

    def submit(
        self,
        model_factory: callable,
        model_packages: str | List[str],
        num_structures: int | "DatasetSize" | "DatasetConfig" = 100,
        checkpoint_name: str | None = None,
        checkpoint_path: str | None = None,
    ):
        """Submit benchmark job to remote executor.

        Args:
            model_factory: User-provided function that takes device and returns an ASE calculator.
                          Example: lambda device: mace_mp(model="medium", device=device)
            model_packages: Python package(s) to install. Can be a single package string
                          (e.g., "mace-torch") or a list (e.g., ["torch", "mace-torch"])
            num_structures: Number of structures to evaluate, or DatasetSize enum, or DatasetConfig
                          (DatasetSize.FULL, DatasetSize.RANDOM_10K, DatasetSize.RANDOM_10K.seed(10))
            checkpoint_name: Optional name for the checkpoint file (e.g. "my_checkpoint.json").
                             If not provided, one will be generated.
            checkpoint_path: Optional path to an existing checkpoint file to resume from.
                             If provided, checkpoint_name is ignored and no new checkpoint is created.
        """
        # Build script with task-specific functions AND user's factory
        script_content = self._build_script(
            self.process_fn,
            self.load_dataset_fn,
            self.calc_metrics_fn,
            model_factory,  # Inject user's factory function
        )

        dependencies = self._prepare_dependencies(model_packages)
        runner_config = self._prepare_runner_config(num_structures)

        # Generate checkpoint name if not provided AND no checkpoint_path is provided
        if not checkpoint_name and not checkpoint_path:
            checkpoint_name = self._generate_checkpoint_name(
                model_packages, runner_config
            )

        if checkpoint_path:
            print(f"Resuming from checkpoint: {checkpoint_path}")
            final_checkpoint_path = checkpoint_path
        else:
            print(
                f"Checkpoint will be saved to: ~/.garden/benchmarks/{checkpoint_name}"
            )
            final_checkpoint_path = f"~/.garden/benchmarks/{checkpoint_name}"

        executor = self.adapter._get_executor()
        future = executor.submit(
            run_remote_benchmark,
            script_content=script_content,
            dependencies=dependencies,
            config=runner_config,
            checkpoint_name=checkpoint_name,
            checkpoint_path=checkpoint_path,
        )

        # Attach checkpoint path to future for programmatic access
        future.checkpoint_path = final_checkpoint_path

        return future

    def local(
        self,
        model_factory: callable,
        model_packages: str | List[str],
        num_structures: int | "DatasetSize" | "DatasetConfig" = 100,
        checkpoint_path: str | None = None,
    ) -> dict:
        """Run benchmark locally.

        Args:
            model_factory: User-provided function that takes device and returns an ASE calculator
            model_packages: Python package(s) to install. Can be a single package string
                          (e.g., "mace-torch") or a list (e.g., ["torch", "mace-torch"])
            num_structures: Number of structures to evaluate, or DatasetSize enum, or DatasetConfig
                          (DatasetSize.FULL, DatasetSize.RANDOM_10K, DatasetSize.RANDOM_10K.seed(10))
            checkpoint_path: Optional path to resume from checkpoint
        """
        from ..utils.remote_execution import run_remote_benchmark

        # Build script with task-specific functions AND user's factory
        script_content = self._build_script(
            self.process_fn, self.load_dataset_fn, self.calc_metrics_fn, model_factory
        )

        dependencies = self._prepare_dependencies(model_packages)
        runner_config = self._prepare_runner_config(num_structures)

        # Run locally (no Globus Compute)
        return run_remote_benchmark(
            script_content=script_content,
            dependencies=dependencies,
            config=runner_config,
            checkpoint_path=checkpoint_path,
        )


class IS2RETask(MatbenchTask):
    """Initial Structure to Relaxed Energy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, task_name="IS2RE", **kwargs)
        self.process_fn = process_batch_relaxation
        self.load_dataset_fn = load_dataset_wbm_initial
        self.calc_metrics_fn = calculate_metrics_energy


class RS2RETask(MatbenchTask):
    """Relaxed Structure to Relaxed Energy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, task_name="RS2RE", **kwargs)
        self.process_fn = process_batch_static
        self.load_dataset_fn = load_dataset_wbm_relaxed
        self.calc_metrics_fn = calculate_metrics_energy


class S2EFSTask(MatbenchTask):
    """Structure to Energy, Forces, Stress."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, task_name="S2EFS", **kwargs)
        self.process_fn = process_batch_forces
        self.load_dataset_fn = load_dataset_mp_trj
        self.calc_metrics_fn = calculate_metrics_forces


class S2EFTask(MatbenchTask):
    """Structure to Energy, Force."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, task_name="S2EF", **kwargs)
        self.process_fn = process_batch_forces
        self.load_dataset_fn = load_dataset_mp_trj
        self.calc_metrics_fn = calculate_metrics_forces


class S2EFSMTask(MatbenchTask):
    """Structure to Energy, Force, Stress, Magmoms."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, task_name="S2EFSM", **kwargs)
        self.process_fn = process_batch_forces
        self.load_dataset_fn = load_dataset_mp_trj
        self.calc_metrics_fn = calculate_metrics_forces


class IS2ETask(MatbenchTask):
    """Initial Structure to Energy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, task_name="IS2E", **kwargs)
        self.process_fn = process_batch_static
        self.load_dataset_fn = load_dataset_wbm_initial
        self.calc_metrics_fn = calculate_metrics_energy


class S2ETask(MatbenchTask):
    """Structure to Energy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, task_name="S2E", **kwargs)
        self.process_fn = process_batch_static
        self.load_dataset_fn = load_dataset_wbm_relaxed
        self.calc_metrics_fn = calculate_metrics_energy


class S2RETask(MatbenchTask):
    """Structure to Relaxed Energy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, task_name="S2RE", **kwargs)
        self.process_fn = process_batch_relaxation
        self.load_dataset_fn = load_dataset_wbm_initial
        self.calc_metrics_fn = calculate_metrics_energy


class RP2RETask(MatbenchTask):
    """Relaxed Prototype to Relaxed Energy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, task_name="RP2RE", **kwargs)
        self.process_fn = process_batch_relaxation
        self.load_dataset_fn = load_dataset_wbm_initial  # Placeholder
        self.calc_metrics_fn = calculate_metrics_energy


class IP2ETask(MatbenchTask):
    """Initial Prototype to Energy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, task_name="IP2E", **kwargs)
        self.process_fn = process_batch_static
        self.load_dataset_fn = load_dataset_wbm_initial  # Placeholder
        self.calc_metrics_fn = calculate_metrics_energy
