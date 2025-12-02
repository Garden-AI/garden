"""Matbench Discovery benchmark task implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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


def load_model(device: str):
    """Initialize the model using the user-provided factory function.

    The factory function is injected into this script by the benchmark framework.
    """
    # Call the user's factory function (injected as load_model_user)
    model = load_model_user(device)  # noqa: F821
    return model


def get_material_ids_for_subset(
    subset_type: str, seed: int = 42
) -> Optional[List[str]]:
    """Get material IDs for a specific dataset subset.

    Args:
        subset_type: One of 'full', 'unique_protos', 'random_10k', 'random_100'
        seed: Random seed for sampling (default: 42)

    Returns:
        List of material IDs, or None for 'full' (load all)
    """
    if subset_type == "full":
        return None  # Load all materials

    import pandas as pd
    from matbench_discovery.data import DataFiles

    # Load wbm_summary
    df = pd.read_csv(DataFiles.wbm_summary.path)

    if subset_type == "unique_protos":
        # Filter to unique prototypes (removes duplicates and MP overlaps)
        df_filtered = df.query("unique_prototype")
        return df_filtered["material_id"].tolist()

    elif subset_type == "random_10k":
        # Random sample of 10k unique prototypes (fixed seed for reproducibility)
        df_filtered = df.query("unique_prototype")
        df_sampled = df_filtered.sample(n=10000, random_state=seed)
        return df_sampled["material_id"].tolist()

    elif subset_type == "random_100":
        # Random sample of 100 unique prototypes (fixed seed for reproducibility)
        # Useful for quick end-to-end testing
        df_filtered = df.query("unique_prototype")
        df_sampled = df_filtered.sample(n=100, random_state=seed)
        return df_sampled["material_id"].tolist()

    else:
        raise ValueError(f"Unknown subset_type: {subset_type}")


# --- Reusable Process Functions ---


def process_batch_relaxation(
    batch_id: int,
    structures: List[Tuple[str, Any]],
    model_config: Dict[str, Any],
    num_threads: int,
) -> Dict[str, Any]:
    """Process a batch of structures for IS2RE (Relaxation)."""
    import logging
    import os
    import time

    import torch
    from ase.optimize import FIRE

    # Configure thread limits to avoid contention
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)

    gpu_id = model_config.get("gpu_id")
    device = setup_device(gpu_id)  # noqa: F821

    worker_logger = logging.getLogger(f"worker_{batch_id}")
    worker_logger.info(
        f"Started relaxation on {device} with {len(structures)} structures. Threads: {num_threads}"
    )

    global _MODEL_CACHE
    try:
        if _MODEL_CACHE is None:
            model = load_model(device)
            _MODEL_CACHE = model
        else:
            model = _MODEL_CACHE
    except Exception as e:
        worker_logger.error(f"Failed to initialize model: {e}")
        worker_logger.error(
            "Model initialization is critical - cannot continue benchmark"
        )
        raise RuntimeError(f"Model initialization failed: {e}") from e

    results = {}
    batch_start = time.time()

    for i, (struct_id, atoms) in enumerate(structures):
        try:
            atoms.calc = model
            opt = FIRE(atoms, logfile=None)
            opt.run(fmax=0.05, steps=500)

            energy = atoms.get_potential_energy()
            results[struct_id] = {"energy": energy}

            if (i + 1) % 10 == 0:
                elapsed = time.time() - batch_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                worker_logger.info(
                    f"Progress: {i + 1}/{len(structures)} ({rate:.2f} struct/s)"
                )

        except Exception as e:
            worker_logger.warning(f"Structure {struct_id} failed: {e}")
            results[struct_id] = {"energy": None, "error": str(e)}

    return results


def load_dataset_wbm_initial(config: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Load initial structures for IS2RE."""
    from io import TextIOWrapper
    from zipfile import ZipFile

    from ase.io import read
    from matbench_discovery.data import DataFiles

    dataset_subset = config.get("dataset_subset", "full")
    dataset_seed = config.get("dataset_seed", 42)
    mat_ids = get_material_ids_for_subset(dataset_subset, seed=dataset_seed)

    structures = []
    zip_path = DataFiles.wbm_initial_atoms.path

    with ZipFile(zip_path, "r") as zf:
        if mat_ids is None:
            # Load all files (full dataset)
            file_list = sorted(
                zf.namelist(),
                key=lambda x: int(x.split(".")[0])
                if x.split(".")[0].isdigit()
                else float("inf"),
            )
            num_structures = config.get("num_structures", 100)
            file_list = file_list[:num_structures]
        else:
            # Filter to specific material IDs
            mat_id_set = set(mat_ids)
            file_list = [
                f for f in zf.namelist() if f.replace(".extxyz", "") in mat_id_set
            ]

        for filename in file_list:
            with zf.open(filename) as f:
                text_stream = TextIOWrapper(f, encoding="utf-8")
                structures.append((filename, read(text_stream, format="extxyz")))
    return structures


def calculate_metrics_energy(
    results: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate energy metrics using matbench-discovery's stable_metrics algorithm.

    Uses the injected stable_metrics function.
    Returns: F1, DAF, Precision, Recall, Accuracy, TPR, FPR, TNR, FNR, TP, FP, TN, FN, MAE, RMSE, R2
    """
    import logging

    import numpy as np

    logger = logging.getLogger("metrics")

    # Results format: {id: {"energy": float, "error": str}}
    if len(results) == 0:
        return {"error": "No results to evaluate"}

    try:
        # Import matbench-discovery data
        from matbench_discovery.data import df_wbm
    except Exception as e:
        return {"error": f"Failed to import matbench-discovery: {e}"}

    # Extract model energies
    model_energies = {}
    for sid, res in results.items():
        if isinstance(res, dict) and res.get("energy") is not None:
            mat_id = sid.replace(".extxyz", "")
            model_energies[mat_id] = res["energy"]

    if not model_energies:
        return {"error": "No valid energies found in results"}

    # Get common IDs between predictions and ground truth
    # Use direct string column names instead of MbdKey enum to avoid issues
    df_wbm_indexed = df_wbm.set_index("material_id")
    common_ids = list(set(model_energies.keys()) & set(df_wbm_indexed.index))

    if not common_ids:
        return {"error": "No matching IDs between results and ground truth"}

    # Get subset of data
    df_subset = df_wbm_indexed.loc[common_ids]

    # Calculate predicted formation energies
    y_pred = np.array([model_energies[mid] for mid in common_ids])
    y_true = df_subset["uncorrected_energy"].values  # Uncorrected total energy
    n_atoms = df_subset["n_sites"].values

    # Predicted formation energy ERROR per atom (from total energy difference)
    # This is the ERROR: (E_pred - E_dft) / n_atoms
    e_form_error = (y_pred - y_true) / n_atoms

    # Get ground truth e_above_hull for stability classification
    each_true = df_subset["e_above_hull_mp2020_corrected_ppd_mp"].values

    # Calculate predicted e_above_hull
    # Since e_form_error is already (E_pred - E_dft)/n_atoms, we just add it to each_true
    each_pred = each_true + e_form_error

    # Debug logging to understand the distribution
    logger.info("Energy statistics:")
    logger.info(
        f"  each_true: min={each_true.min():.4f}, max={each_true.max():.4f}, mean={each_true.mean():.4f}"
    )
    logger.info(
        f"  each_pred: min={each_pred.min():.4f}, max={each_pred.max():.4f}, mean={each_pred.mean():.4f}"
    )

    # Calculate global prevalence for DAF normalization (matches official leaderboard)
    # Filter to unique prototypes
    df_unique = df_wbm.query("unique_prototype")
    # Calculate prevalence: (stable count) / (total count)
    # Stability threshold is 0.0
    stable_count = (df_unique["e_above_hull_mp2020_corrected_ppd_mp"] <= 0).sum()
    global_prevalence = stable_count / len(df_unique)

    logger.info(
        f"Using global prevalence for DAF: {global_prevalence:.6f} ({stable_count}/{len(df_unique)})"
    )

    # Calculate metrics using the injected function
    # stable_metrics is injected into the script scope
    metrics = stable_metrics(each_true, each_pred, prevalence=global_prevalence)

    # Add num_evaluated
    metrics["num_evaluated"] = len(common_ids)

    return metrics


def process_batch_static(
    batch_id: int,
    structures: List[Tuple[str, Any]],
    model_config: Dict[str, Any],
    num_threads: int,
) -> Dict[str, Any]:
    """Process a batch of structures for RS2RE (Static Calculation)."""
    import logging
    import os
    import time

    import torch

    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)

    gpu_id = model_config.get("gpu_id")
    device = setup_device(gpu_id)  # noqa: F821

    worker_logger = logging.getLogger(f"worker_{batch_id}")
    worker_logger.info(
        f"Started static calculation on {device} with {len(structures)} structures."
    )

    global _MODEL_CACHE
    try:
        if _MODEL_CACHE is None:
            model = load_model(device)
            _MODEL_CACHE = model
        else:
            model = _MODEL_CACHE
    except Exception as e:
        return {sid: {"energy": None, "error": str(e)} for sid, _ in structures}

    results = {}
    batch_start = time.time()

    for i, (struct_id, atoms) in enumerate(structures):
        try:
            atoms.calc = model
            # No relaxation, just static energy
            energy = atoms.get_potential_energy()
            results[struct_id] = {"energy": energy}

            if (i + 1) % 50 == 0:
                elapsed = time.time() - batch_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                worker_logger.info(
                    f"Progress: {i + 1}/{len(structures)} ({rate:.2f} struct/s)"
                )

        except Exception as e:
            worker_logger.warning(f"Structure {struct_id} failed: {e}")
            results[struct_id] = {"energy": None, "error": str(e)}

    return results


def load_dataset_wbm_relaxed(config: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Load relaxed structures for RS2RE."""
    from io import TextIOWrapper
    from zipfile import ZipFile

    from ase.io import read
    from matbench_discovery.data import DataFiles

    dataset_subset = config.get("dataset_subset", "full")
    dataset_seed = config.get("dataset_seed", 42)
    mat_ids = get_material_ids_for_subset(dataset_subset, seed=dataset_seed)

    structures = []
    # Use relaxed atoms
    zip_path = DataFiles.wbm_relaxed_atoms.path

    with ZipFile(zip_path, "r") as zf:
        if mat_ids is None:
            # Load all files (full dataset)
            file_list = sorted(
                zf.namelist(),
                key=lambda x: int(x.split(".")[0])
                if x.split(".")[0].isdigit()
                else float("inf"),
            )
            num_structures = config.get("num_structures", 100)
            file_list = file_list[:num_structures]
        else:
            # Filter to specific material IDs
            mat_id_set = set(mat_ids)
            file_list = [
                f for f in zf.namelist() if f.replace(".extxyz", "") in mat_id_set
            ]

        for filename in file_list:
            with zf.open(filename) as f:
                text_stream = TextIOWrapper(f, encoding="utf-8")
                structures.append((filename, read(text_stream, format="extxyz")))
    return structures


# Reuse calculate_metrics_energy for all energy-only tasks


def process_batch_forces(
    batch_id: int,
    structures: List[Tuple[str, Any]],
    model_config: Dict[str, Any],
    num_threads: int,
) -> Dict[str, Any]:
    """Process a batch of structures for S2EFS (Energy, Forces, Stress)."""
    import logging
    import os
    import time

    import torch

    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)

    gpu_id = model_config.get("gpu_id")
    device = setup_device(gpu_id)  # noqa: F821

    worker_logger = logging.getLogger(f"worker_{batch_id}")
    worker_logger.info(
        f"Started forces calculation on {device} with {len(structures)} structures."
    )

    global _MODEL_CACHE
    try:
        if _MODEL_CACHE is None:
            model = load_model(device)
            _MODEL_CACHE = model
        else:
            model = _MODEL_CACHE
    except Exception as e:
        return {sid: {"error": str(e)} for sid, _ in structures}

    results = {}
    batch_start = time.time()

    for i, (struct_id, atoms) in enumerate(structures):
        try:
            atoms.calc = model

            energy = atoms.get_potential_energy()
            forces = atoms.get_forces().tolist()
            stress = atoms.get_stress().tolist()

            results[struct_id] = {"energy": energy, "forces": forces, "stress": stress}

            if (i + 1) % 50 == 0:
                elapsed = time.time() - batch_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                worker_logger.info(
                    f"Progress: {i + 1}/{len(structures)} ({rate:.2f} struct/s)"
                )

        except Exception as e:
            worker_logger.warning(f"Structure {struct_id} failed: {e}")
            results[struct_id] = {"error": str(e)}

    return results


def load_dataset_mp_trj(config: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Load MP trajectories for S2EFS."""
    from io import TextIOWrapper
    from zipfile import ZipFile

    from ase.io import read
    from matbench_discovery.data import DataFiles

    num_structures = config.get("num_structures", 100)
    structures = []
    # Use MP trajectories
    zip_path = DataFiles.mp_trj_extxyz.path

    with ZipFile(zip_path, "r") as zf:
        file_list = sorted(zf.namelist())
        for filename in file_list[:num_structures]:
            with zf.open(filename) as f:
                text_stream = TextIOWrapper(f, encoding="utf-8")
                # Read all frames? Or just one? Usually S2EFS is on frames.
                # Let's assume we evaluate on the last frame or all frames.
                # For simplicity, let's take the last frame (relaxed?) or random?
                # Actually, MP trj contains relaxation steps.
                # Let's read the last frame for now as a proxy for "a structure".
                # Or better, read all frames and treat them as separate tasks?
                # For this benchmark, let's just treat the file as containing one structure per file if possible,
                # or just take the last one.
                atoms_list = read(text_stream, format="extxyz", index=":")
                if atoms_list:
                    # Just take the last one for now
                    structures.append((filename, atoms_list[-1]))
    return structures


def calculate_metrics_forces(
    results: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate comprehensive S2EFS metrics (Energy, Forces, Stress).

    Returns MAE, RMSE, and R2 for each component.
    """
    from io import TextIOWrapper
    from zipfile import ZipFile

    import numpy as np
    from ase.io import read
    from matbench_discovery.data import DataFiles
    from sklearn.metrics import r2_score

    # We need to load ground truth from the dataset itself because MP trj has E/F/S in the extxyz
    # This is expensive to re-read. Ideally we should have passed GT in results or loaded it efficiently.
    # For now, let's re-read the GT for the processed IDs.

    metrics = {
        "energy_mae": [],
        "energy_rmse": [],
        "force_mae": [],
        "force_rmse": [],
        "stress_mae": [],
        "stress_rmse": [],
    }

    # Collect all predictions and ground truth for R2 calculation
    all_e_pred, all_e_true = [], []
    all_f_pred, all_f_true = [], []
    all_s_pred, all_s_true = [], []

    zip_path = DataFiles.mp_trj_extxyz.path

    with ZipFile(zip_path, "r") as zf:
        for sid, res in results.items():
            if "error" in res:
                continue

            try:
                with zf.open(sid) as f:
                    text_stream = TextIOWrapper(f, encoding="utf-8")
                    atoms_list = read(text_stream, format="extxyz", index=":")
                    gt_atoms = atoms_list[-1]  # Matching load_dataset logic

                    # Energy (per atom)
                    e_pred = res["energy"]
                    e_true = gt_atoms.get_potential_energy()
                    n_atoms = len(gt_atoms)

                    energy_error = abs(e_pred - e_true) / n_atoms
                    metrics["energy_mae"].append(energy_error)
                    metrics["energy_rmse"].append(energy_error**2)

                    all_e_pred.append(e_pred / n_atoms)
                    all_e_true.append(e_true / n_atoms)

                    # Forces
                    f_pred = np.array(res["forces"])
                    f_true = gt_atoms.get_forces()
                    force_error = np.abs(f_pred - f_true)
                    metrics["force_mae"].append(force_error.mean())
                    metrics["force_rmse"].append((force_error**2).mean())

                    all_f_pred.extend(f_pred.flatten())
                    all_f_true.extend(f_true.flatten())

                    # Stress
                    s_pred = np.array(res["stress"])
                    s_true = gt_atoms.get_stress()
                    stress_error = np.abs(s_pred - s_true)
                    metrics["stress_mae"].append(stress_error.mean())
                    metrics["stress_rmse"].append((stress_error**2).mean())

                    all_s_pred.extend(s_pred.flatten())
                    all_s_true.extend(s_true.flatten())

            except Exception:
                pass

    # Calculate final metrics
    result_metrics = {}

    if metrics["energy_mae"]:
        result_metrics["energy_mae"] = float(np.mean(metrics["energy_mae"]))
        result_metrics["energy_rmse"] = float(np.sqrt(np.mean(metrics["energy_rmse"])))
        result_metrics["energy_r2"] = (
            float(r2_score(all_e_true, all_e_pred))
            if len(all_e_true) > 1
            else float("nan")
        )

    if metrics["force_mae"]:
        result_metrics["force_mae"] = float(np.mean(metrics["force_mae"]))
        result_metrics["force_rmse"] = float(np.sqrt(np.mean(metrics["force_rmse"])))
        result_metrics["force_r2"] = (
            float(r2_score(all_f_true, all_f_pred))
            if len(all_f_true) > 1
            else float("nan")
        )

    if metrics["stress_mae"]:
        result_metrics["stress_mae"] = float(np.mean(metrics["stress_mae"]))
        result_metrics["stress_rmse"] = float(np.sqrt(np.mean(metrics["stress_rmse"])))
        result_metrics["stress_r2"] = (
            float(r2_score(all_s_true, all_s_pred))
            if len(all_s_true) > 1
            else float("nan")
        )

    result_metrics["num_evaluated"] = len(metrics["energy_mae"])

    return result_metrics


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
        builder.add_import("from typing import List, Dict, Any, Tuple, Optional")
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

        # Add task-specific functions with standard names expected by runner
        builder.add_function(process_fn, name="process_batch")
        builder.add_function(load_dataset_fn, name="load_dataset")
        builder.add_function(calc_metrics_fn, name="calculate_metrics_remote")

        # Inject metrics helper functions
        builder.add_function(classify_stable)
        builder.add_function(stable_metrics)

        return builder.build()

    def submit(
        self,
        model_factory: callable,
        model_packages: str | List[str],
        num_structures: int | "DatasetSize" | "DatasetConfig" = 100,
        checkpoint_name: str | None = None,
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
        """
        import time
        import uuid

        from .enums import DatasetConfig, DatasetSize

        # Build script with task-specific functions AND user's factory
        script_content = self._build_script(
            self.process_fn,
            self.load_dataset_fn,
            self.calc_metrics_fn,
            model_factory,  # Inject user's factory function
        )

        # Handle single package string or list of packages
        packages = (
            [model_packages] if isinstance(model_packages, str) else model_packages
        )
        dependencies = ["matbench-discovery>=1.3.0"] + packages

        # Handle DatasetSize enum, DatasetConfig, or integer
        if isinstance(num_structures, DatasetSize):
            runner_config = {
                "repo_url": self.repo_url,
                "repo_ref": self.repo_ref,
                "dataset_subset": num_structures.value,
                "dataset_seed": 42,  # Default seed
            }
        elif isinstance(num_structures, DatasetConfig):
            runner_config = {
                "repo_url": self.repo_url,
                "repo_ref": self.repo_ref,
                "dataset_subset": num_structures.subset.value,
                "dataset_seed": num_structures.seed,
            }
        else:
            # Integer - use traditional num_structures approach
            runner_config = {
                "repo_url": self.repo_url,
                "repo_ref": self.repo_ref,
                "num_structures": num_structures,
                "dataset_subset": "full",
            }

        # Generate checkpoint name if not provided
        if not checkpoint_name:
            # Format: matbench_{model}_{subset}_{timestamp}_{uuid}.json
            # Clean up model name for filename
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
            checkpoint_name = (
                f"matbench_{model_str}_{subset_str}_{timestamp}_{short_uuid}.json"
            )

        print(f"Checkpoint will be saved to: ~/.garden/benchmarks/{checkpoint_name}")

        executor = self.adapter._get_executor()
        future = executor.submit(
            run_remote_benchmark,
            script_content=script_content,
            dependencies=dependencies,
            config=runner_config,
            checkpoint_name=checkpoint_name,
        )

        # Attach checkpoint path to future for programmatic access
        future.checkpoint_path = f"~/.garden/benchmarks/{checkpoint_name}"

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
        from .enums import DatasetConfig, DatasetSize

        # Build script with task-specific functions AND user's factory
        script_content = self._build_script(
            self.process_fn, self.load_dataset_fn, self.calc_metrics_fn, model_factory
        )

        # Handle single package string or list of packages
        packages = (
            [model_packages] if isinstance(model_packages, str) else model_packages
        )
        dependencies = ["matbench-discovery>=1.3.0"] + packages

        # Handle DatasetSize enum, DatasetConfig, or integer
        if isinstance(num_structures, DatasetSize):
            runner_config = {
                "repo_url": self.repo_url,
                "repo_ref": self.repo_ref,
                "dataset_subset": num_structures.value,
                "dataset_seed": 42,  # Default seed
            }
        elif isinstance(num_structures, DatasetConfig):
            runner_config = {
                "repo_url": self.repo_url,
                "repo_ref": self.repo_ref,
                "dataset_subset": num_structures.subset.value,
                "dataset_seed": num_structures.seed,
            }
        else:
            # Integer - use traditional num_structures approach
            runner_config = {
                "repo_url": self.repo_url,
                "repo_ref": self.repo_ref,
                "num_structures": num_structures,
                "dataset_subset": "full",
            }

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
