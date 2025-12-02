"""Remote functions for Matbench Discovery benchmark.

These functions are injected into the remote script.
They must be self-contained (imports inside or provided by builder).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

# ------------------------------------------------------------------------------
# Common Helpers
# ------------------------------------------------------------------------------


def _process_batch_common(
    batch_id: int,
    structures: List[Tuple[str, Any]],
    model_config: Dict[str, Any],
    num_threads: int,
    compute_fn: Callable[[Any, Any], Dict[str, Any]],
    task_name: str,
) -> Dict[str, Any]:
    """Common logic for processing a batch of structures.

    Args:
        batch_id: ID of the current batch
        structures: List of (id, atoms) tuples
        model_config: Configuration for the model
        num_threads: Number of threads to use
        compute_fn: Function taking (model, atoms) and returning a result dict
        task_name: Name of the task for logging
    """
    import logging
    import os
    import time

    import torch

    # Configure thread limits to avoid contention
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)

    gpu_id = model_config.get("gpu_id")
    device = setup_device(gpu_id)  # noqa: F821

    worker_logger = logging.getLogger(f"worker_{batch_id}")
    worker_logger.info(
        f"Started {task_name} on {device} with {len(structures)} structures. Threads: {num_threads}"
    )

    global _MODEL_CACHE
    try:
        if _MODEL_CACHE is None:
            model = load_model(device)  # noqa: F821
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
            # Run the specific computation
            result = compute_fn(model, atoms)
            results[struct_id] = result

            if (i + 1) % 10 == 0:
                elapsed = time.time() - batch_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                worker_logger.info(
                    f"Progress: {i + 1}/{len(structures)} ({rate:.2f} struct/s)"
                )

        except Exception as e:
            worker_logger.warning(f"Structure {struct_id} failed: {e}")
            results[struct_id] = {"error": str(e)}

    return results


def _load_dataset_common(
    config: Dict[str, Any],
    zip_path: str,
    read_format: str = "extxyz",
    read_index: str | slice = None,
) -> List[Tuple[str, Any]]:
    """Common logic for loading datasets from a zip file."""
    from io import TextIOWrapper
    from zipfile import ZipFile

    from ase.io import read

    # get_material_ids_for_subset is injected
    dataset_subset = config.get("dataset_subset", "full")
    dataset_seed = config.get("dataset_seed", 42)
    mat_ids = get_material_ids_for_subset(dataset_subset, seed=dataset_seed)  # noqa: F821

    structures = []

    with ZipFile(zip_path, "r") as zf:
        if mat_ids is None:
            # Load all files (full dataset)
            # Sort by numeric ID if possible
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
                if read_index is not None:
                    atoms_list = read(text_stream, format=read_format, index=read_index)
                    # If we got a list and need one item, take the last one (common for trajectories)
                    if isinstance(atoms_list, list) and atoms_list:
                        structures.append((filename, atoms_list[-1]))
                    elif not isinstance(atoms_list, list):
                        structures.append((filename, atoms_list))
                else:
                    structures.append((filename, read(text_stream, format=read_format)))

    return structures


# ------------------------------------------------------------------------------
# Injected Functions
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


def process_batch_relaxation(
    batch_id: int,
    structures: List[Tuple[str, Any]],
    model_config: Dict[str, Any],
    num_threads: int,
) -> Dict[str, Any]:
    """Process a batch of structures for IS2RE (Relaxation)."""
    from ase.optimize import FIRE

    def compute(model, atoms):
        atoms.calc = model
        opt = FIRE(atoms, logfile=None)
        opt.run(fmax=0.05, steps=500)
        energy = atoms.get_potential_energy()
        return {"energy": energy}

    return _process_batch_common(
        batch_id, structures, model_config, num_threads, compute, "relaxation"
    )


def process_batch_static(
    batch_id: int,
    structures: List[Tuple[str, Any]],
    model_config: Dict[str, Any],
    num_threads: int,
) -> Dict[str, Any]:
    """Process a batch of structures for RS2RE (Static Calculation)."""

    def compute(model, atoms):
        atoms.calc = model
        energy = atoms.get_potential_energy()
        return {"energy": energy}

    return _process_batch_common(
        batch_id, structures, model_config, num_threads, compute, "static calculation"
    )


def process_batch_forces(
    batch_id: int,
    structures: List[Tuple[str, Any]],
    model_config: Dict[str, Any],
    num_threads: int,
) -> Dict[str, Any]:
    """Process a batch of structures for S2EFS (Energy, Forces, Stress)."""

    def compute(model, atoms):
        atoms.calc = model
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces().tolist()
        stress = atoms.get_stress().tolist()
        return {"energy": energy, "forces": forces, "stress": stress}

    return _process_batch_common(
        batch_id, structures, model_config, num_threads, compute, "forces calculation"
    )


def load_dataset_wbm_initial(config: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Load initial structures for IS2RE."""
    from matbench_discovery.data import DataFiles

    return _load_dataset_common(config, DataFiles.wbm_initial_atoms.path)


def load_dataset_wbm_relaxed(config: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Load relaxed structures for RS2RE."""
    from matbench_discovery.data import DataFiles

    return _load_dataset_common(config, DataFiles.wbm_relaxed_atoms.path)


def load_dataset_mp_trj(config: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Load MP trajectories for S2EFS."""
    from matbench_discovery.data import DataFiles

    # Use index=":" to read all frames, but _load_dataset_common handles taking the last one
    return _load_dataset_common(config, DataFiles.mp_trj_extxyz.path, read_index=":")


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
    metrics = stable_metrics(each_true, each_pred, prevalence=global_prevalence)  # noqa: F821

    # Add num_evaluated
    metrics["num_evaluated"] = len(common_ids)

    return metrics


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
