# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "groundhog-hpc",
#     "garden-ai",
#     "ase",
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "torch",
#     "matbench-discovery",
# ]
# ///
"""Matbench Discovery benchmark task implementations using Groundhog HPC."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import multiprocessing
import os
import sys
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

import groundhog_hpc as hog
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# Ensure local modules can be imported during local execution
sys.path.append(os.getcwd())

if TYPE_CHECKING:
    from .enums import DatasetConfig, DatasetSize

# ------------------------------------------------------------------------------
# BOILERPLATE: Logging & Device Setup
# ------------------------------------------------------------------------------


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [PID:%(process)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger("benchmark_runner")


def setup_device(gpu_id: Optional[int] = None) -> str:
    """Setup compute device for this process."""
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{gpu_id}" if gpu_id is not None else "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np

    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


# ------------------------------------------------------------------------------
# METRICS HELPERS (Inlined from metrics.py)
# ------------------------------------------------------------------------------


def classify_stable(
    each_true: Sequence[float] | pd.Series | np.ndarray,
    each_pred: Sequence[float] | pd.Series | np.ndarray,
    *,
    stability_threshold: float = 0.0,
    fillna: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if len(each_true) != len(each_pred):
        raise ValueError(f"{len(each_true)=} != {len(each_pred)=}")

    each_true_arr, each_pred_arr = pd.Series(each_true), pd.Series(each_pred)

    if stability_threshold is None or np.isnan(stability_threshold):
        raise ValueError("stability_threshold must be a real number")
    actual_pos = each_true_arr <= (stability_threshold or 0)
    actual_neg = each_true_arr > (stability_threshold or 0)

    model_pos = each_pred_arr <= (stability_threshold or 0)
    model_neg = each_pred_arr > (stability_threshold or 0)

    if fillna:
        nan_mask = np.isnan(each_pred)
        model_pos[nan_mask] = False
        model_neg[nan_mask] = True

        n_pos, n_neg, total = model_pos.sum(), model_neg.sum(), len(each_pred)
        if n_pos + n_neg != total:
            raise ValueError(
                f"after filling NaNs, the sum of positive ({n_pos}) and negative "
                f"({n_neg}) predictions should add up to {total=}"
            )

    true_pos = actual_pos & model_pos
    false_neg = actual_pos & model_neg
    false_pos = actual_neg & model_pos
    true_neg = actual_neg & model_neg

    return true_pos, false_neg, false_pos, true_neg


def stable_metrics(
    each_true: Sequence[float] | pd.Series | np.ndarray,
    each_pred: Sequence[float] | pd.Series | np.ndarray,
    *,
    stability_threshold: float = 0.0,
    fillna: bool = True,
    prevalence: float | None = None,
) -> dict[str, float]:
    n_true_pos, n_false_neg, n_false_pos, n_true_neg = map(
        sum,
        classify_stable(
            each_true, each_pred, stability_threshold=stability_threshold, fillna=fillna
        ),
    )

    n_total_pos = n_true_pos + n_false_neg
    n_total_neg = n_true_neg + n_false_pos
    if prevalence is None:
        prevalence = (
            n_total_pos / (n_total_pos + n_total_neg)
            if (n_total_pos + n_total_neg) > 0
            else float("nan")
        )
    precision = (
        n_true_pos / (n_true_pos + n_false_pos)
        if (n_true_pos + n_false_pos) > 0
        else float("nan")
    )
    recall = n_true_pos / n_total_pos if n_total_pos > 0 else float("nan")

    TPR = recall
    FPR = n_false_pos / n_total_neg if n_total_neg > 0 else float("nan")
    TNR = n_true_neg / n_total_neg if n_total_neg > 0 else float("nan")
    FNR = n_false_neg / n_total_pos if n_total_pos > 0 else float("nan")

    if FPR > 0 and TNR > 0 and FPR + TNR != 1:
        if abs(FPR + TNR - 1) > 1e-6:
            raise ValueError(f"{FPR=} {TNR=} don't add up to 1")

    if TPR > 0 and FNR > 0 and TPR + FNR != 1:
        if abs(TPR + FNR - 1) > 1e-6:
            raise ValueError(f"{TPR=} {FNR=} don't add up to 1")

    is_nan = np.isnan(each_true) | np.isnan(each_pred)
    each_true, each_pred = np.array(each_true)[~is_nan], np.array(each_pred)[~is_nan]

    if precision + recall == 0:
        f1_score = float("nan")
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return dict(
        F1=f1_score,
        DAF=precision / prevalence if prevalence > 0 else float("nan"),
        Precision=precision,
        Recall=recall,
        Accuracy=(
            (n_true_pos + n_true_neg) / (n_total_pos + n_total_neg)
            if (n_total_pos + n_total_neg > 0)
            else float("nan")
        ),
        TPR=TPR,
        FPR=FPR,
        TNR=TNR,
        FNR=FNR,
        TP=n_true_pos,
        FP=n_false_pos,
        TN=n_true_neg,
        FN=n_false_neg,
        MAE=np.abs(each_true - each_pred).mean(),
        RMSE=((each_true - each_pred) ** 2).mean() ** 0.5,
        R2=r2_score(each_true, each_pred) if len(each_true) > 1 else float("nan"),
    )


# ------------------------------------------------------------------------------
# REMOTE HELPERS (Inlined from remote.py)
# ------------------------------------------------------------------------------

_MODEL_CACHE = None


def _process_batch_common(
    batch_id: int,
    structures: List[Any],
    model_config: Dict[str, Any],
    num_threads: int,
    compute_fn: Callable[[Any, Any], Dict[str, Any]],
    task_name: str,
    model_factory: Callable[[str], Any],
) -> Dict[str, Any]:
    import logging
    import os
    import time

    import torch

    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)

    gpu_id = model_config.get("gpu_id")
    device = setup_device(gpu_id)

    worker_logger = logging.getLogger(f"worker_{batch_id}")
    worker_logger.info(
        f"Started {task_name} on {device} with {len(structures)} structures. Threads: {num_threads}"
    )

    global _MODEL_CACHE
    try:
        if _MODEL_CACHE is None:
            model = model_factory(device)
            _MODEL_CACHE = model
        else:
            model = _MODEL_CACHE
    except Exception as e:
        worker_logger.error(f"Failed to initialize model: {e}")
        raise RuntimeError(f"Model initialization failed: {e}") from e

    results = {}
    batch_start = time.time()

    for i, (struct_id, atoms) in enumerate(structures):
        try:
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


def get_material_ids_for_subset(
    subset_type: str, seed: int = 42
) -> Optional[List[str]]:
    if subset_type == "full":
        return None

    import pandas as pd
    from matbench_discovery.data import DataFiles

    df = pd.read_csv(DataFiles.wbm_summary.path)

    if subset_type == "unique_protos":
        df_filtered = df.query("unique_prototype")
        return df_filtered["material_id"].tolist()

    elif subset_type == "random_10k":
        df_filtered = df.query("unique_prototype")
        df_sampled = df_filtered.sample(n=10000, random_state=seed)
        return df_sampled["material_id"].tolist()

    elif subset_type == "random_100":
        df_filtered = df.query("unique_prototype")
        df_sampled = df_filtered.sample(n=100, random_state=seed)
        return df_sampled["material_id"].tolist()

    else:
        raise ValueError(f"Unknown subset_type: {subset_type}")


def _load_dataset_common(
    config: Dict[str, Any],
    zip_path: str,
    read_format: str = "extxyz",
    read_index: str | slice = None,
) -> List[Any]:
    from io import TextIOWrapper
    from zipfile import ZipFile

    from ase.io import read

    dataset_subset = config.get("dataset_subset", "full")
    dataset_seed = config.get("dataset_seed", 42)
    mat_ids = get_material_ids_for_subset(dataset_subset, seed=dataset_seed)

    structures = []

    with ZipFile(zip_path, "r") as zf:
        if mat_ids is None:
            file_list = sorted(
                zf.namelist(),
                key=lambda x: int(x.split(".")[0])
                if x.split(".")[0].isdigit()
                else float("inf"),
            )
            num_structures = config.get("num_structures", 100)
            if isinstance(num_structures, int):
                file_list = file_list[:num_structures]
        else:
            mat_id_set = set(mat_ids)
            file_list = [
                f for f in zf.namelist() if f.replace(".extxyz", "") in mat_id_set
            ]

        for filename in file_list:
            with zf.open(filename) as f:
                text_stream = TextIOWrapper(f, encoding="utf-8")
                if read_index is not None:
                    atoms_list = read(text_stream, format=read_format, index=read_index)
                    if isinstance(atoms_list, list) and atoms_list:
                        structures.append((filename, atoms_list[-1]))
                    elif not isinstance(atoms_list, list):
                        structures.append((filename, atoms_list))
                else:
                    structures.append((filename, read(text_stream, format=read_format)))

    return structures


# Task-specific helpers
def process_batch_relaxation(
    batch_id: int,
    structures: List[Any],
    model_config: Dict[str, Any],
    num_threads: int,
    model_factory: Callable[[str], Any],
) -> Dict[str, Any]:
    from ase.optimize import FIRE

    def compute(model, atoms):
        atoms.calc = model
        opt = FIRE(atoms, logfile=None)
        opt.run(fmax=0.05, steps=500)
        energy = atoms.get_potential_energy()
        return {"energy": energy}

    return _process_batch_common(
        batch_id,
        structures,
        model_config,
        num_threads,
        compute,
        "relaxation",
        model_factory,
    )


def process_batch_static(
    batch_id: int,
    structures: List[Any],
    model_config: Dict[str, Any],
    num_threads: int,
    model_factory: Callable[[str], Any],
) -> Dict[str, Any]:
    def compute(model, atoms):
        atoms.calc = model
        energy = atoms.get_potential_energy()
        return {"energy": energy}

    return _process_batch_common(
        batch_id,
        structures,
        model_config,
        num_threads,
        compute,
        "static calculation",
        model_factory,
    )


def process_batch_forces(
    batch_id: int,
    structures: List[Any],
    model_config: Dict[str, Any],
    num_threads: int,
    model_factory: Callable[[str], Any],
) -> Dict[str, Any]:
    def compute(model, atoms):
        atoms.calc = model
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces().tolist()
        stress = atoms.get_stress().tolist()
        return {"energy": energy, "forces": forces, "stress": stress}

    return _process_batch_common(
        batch_id,
        structures,
        model_config,
        num_threads,
        compute,
        "forces calculation",
        model_factory,
    )


def load_dataset_wbm_initial(config: Dict[str, Any]) -> List[Any]:
    from matbench_discovery.data import DataFiles

    return _load_dataset_common(config, DataFiles.wbm_initial_atoms.path)


def load_dataset_wbm_relaxed(config: Dict[str, Any]) -> List[Any]:
    from matbench_discovery.data import DataFiles

    return _load_dataset_common(config, DataFiles.wbm_relaxed_atoms.path)


def load_dataset_mp_trj(config: Dict[str, Any]) -> List[Any]:
    from matbench_discovery.data import DataFiles

    return _load_dataset_common(config, DataFiles.mp_trj_extxyz.path, read_index=":")


def calculate_metrics_energy(
    results: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    import numpy as np
    from matbench_discovery.data import df_wbm

    if len(results) == 0:
        return {"error": "No results to evaluate"}

    model_energies = {}
    for sid, res in results.items():
        if isinstance(res, dict) and res.get("energy") is not None:
            mat_id = sid.replace(".extxyz", "")
            model_energies[mat_id] = res["energy"]

    if not model_energies:
        return {"error": "No valid energies found in results"}

    df_wbm_indexed = df_wbm.set_index("material_id")
    common_ids = list(set(model_energies.keys()) & set(df_wbm_indexed.index))

    if not common_ids:
        return {"error": "No matching IDs between results and ground truth"}

    df_subset = df_wbm_indexed.loc[common_ids]
    y_pred = np.array([model_energies[mid] for mid in common_ids])
    y_true = df_subset["uncorrected_energy"].values
    n_atoms = df_subset["n_sites"].values

    e_form_error = (y_pred - y_true) / n_atoms
    each_true = df_subset["e_above_hull_mp2020_corrected_ppd_mp"].values
    each_pred = each_true + e_form_error

    df_unique = df_wbm.query("unique_prototype")
    stable_count = (df_unique["e_above_hull_mp2020_corrected_ppd_mp"] <= 0).sum()
    global_prevalence = stable_count / len(df_unique)

    metrics = stable_metrics(each_true, each_pred, prevalence=global_prevalence)
    metrics["num_evaluated"] = len(common_ids)
    return metrics


def calculate_metrics_forces(
    results: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    from io import TextIOWrapper
    from zipfile import ZipFile

    import numpy as np
    from ase.io import read
    from matbench_discovery.data import DataFiles
    from sklearn.metrics import r2_score

    metrics = {
        "energy_mae": [],
        "energy_rmse": [],
        "force_mae": [],
        "force_rmse": [],
        "stress_mae": [],
        "stress_rmse": [],
    }
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
                    gt_atoms = atoms_list[-1]

                    e_pred = res["energy"]
                    e_true = gt_atoms.get_potential_energy()
                    n_atoms = len(gt_atoms)
                    energy_error = abs(e_pred - e_true) / n_atoms
                    metrics["energy_mae"].append(energy_error)
                    metrics["energy_rmse"].append(energy_error**2)
                    all_e_pred.append(e_pred / n_atoms)
                    all_e_true.append(e_true / n_atoms)

                    f_pred = np.array(res["forces"])
                    f_true = gt_atoms.get_forces()
                    force_error = np.abs(f_pred - f_true)
                    metrics["force_mae"].append(force_error.mean())
                    metrics["force_rmse"].append((force_error**2).mean())
                    all_f_pred.extend(f_pred.flatten())
                    all_f_true.extend(f_true.flatten())

                    s_pred = np.array(res["stress"])
                    s_true = gt_atoms.get_stress()
                    stress_error = np.abs(s_pred - s_true)
                    metrics["stress_mae"].append(stress_error.mean())
                    metrics["stress_rmse"].append((stress_error**2).mean())
                    all_s_pred.extend(s_pred.flatten())
                    all_s_true.extend(s_true.flatten())

            except Exception:
                pass

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
# MAIN RUNNER (Inlined from runners.py)
# ------------------------------------------------------------------------------


def run_benchmark_hog(
    config: Dict[str, Any],
    model_factory: Any,
    load_dataset_fn: Any,
    process_fn: Any,
    calc_metrics_fn: Any,
) -> Dict[str, Any]:
    logger = setup_logging()
    logger.info("Starting benchmark runner...")

    checkpoint_path = config.get("checkpoint_path")
    results = {}

    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            with open(checkpoint_path) as f:
                results = json.load(f)
            logger.info(f"Found {len(results)} processed items in checkpoint")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")

    try:
        all_items = load_dataset_fn(config)
        logger.info(f"Loaded {len(all_items)} total items")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        import traceback

        traceback.print_exc()
        raise

    items_to_process = [
        (item_id, item) for item_id, item in all_items if str(item_id) not in results
    ]

    if not items_to_process:
        logger.info("All items already processed!")
        return {"results": results, "metrics": {}}

    logger.info(f"Processing {len(items_to_process)} remaining items")

    import random

    random.seed(42)
    random.shuffle(items_to_process)

    try:
        import torch

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        num_gpus = 0

    use_multi_gpu = config.get("use_multi_gpu", True) and num_gpus > 1
    total_cores = os.cpu_count() or 1
    num_workers = num_gpus if use_multi_gpu else 1
    available_cores = max(1, total_cores - 2) if total_cores > 4 else total_cores
    threads_per_worker = max(1, available_cores // num_workers)

    logger.info(
        f"Resources: {num_gpus} GPUs, {total_cores} Cores. Using {num_workers} workers ({threads_per_worker} threads/worker)"
    )

    start_time = time.time()
    chunk_size = 1000 * num_workers
    chunks = [
        items_to_process[i : i + chunk_size]
        for i in range(0, len(items_to_process), chunk_size)
    ]

    ctx = multiprocessing.get_context("spawn")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers, mp_context=ctx
    ) as executor:
        for chunk_idx, chunk in enumerate(chunks):
            chunk_start = time.time()
            logger.info(
                f"Starting chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} items)"
            )

            futures = []
            batch_size = (len(chunk) + num_workers - 1) // num_workers

            for i in range(num_workers):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(chunk))
                if start < end:
                    batch = chunk[start:end]
                    worker_config = config.copy()
                    worker_config["gpu_id"] = i if use_multi_gpu else None
                    futures.append(
                        executor.submit(
                            process_fn,
                            i,
                            batch,
                            worker_config,
                            threads_per_worker,
                            model_factory,
                        )
                    )

            chunk_results = {}
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_res = future.result()
                    chunk_results.update(batch_res)
                except Exception as e:
                    logger.error(f"Worker failed in chunk {chunk_idx}: {e}")
                    raise RuntimeError(
                        "Aborting benchmark due to worker failure"
                    ) from e

            results.update(chunk_results)

            if checkpoint_path:
                try:
                    tmp_path = checkpoint_path + ".tmp"
                    with open(tmp_path, "w") as f:
                        clean_results = convert_numpy_types(results)
                        json.dump(clean_results, f, indent=2)
                    os.replace(tmp_path, checkpoint_path)
                    logger.info(f"Checkpoint saved to {checkpoint_path}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint: {e}")

            elapsed = time.time() - chunk_start
            logger.info(f"Chunk {chunk_idx + 1} complete in {elapsed:.1f}s")

    total_elapsed = time.time() - start_time
    logger.info(f"Benchmark complete in {total_elapsed:.1f}s.")

    logger.info("Calculating metrics...")
    try:
        metrics = calc_metrics_fn(results, config)
        logger.info(f"Metrics calculated: {metrics}")
    except Exception as e:
        logger.error(f"Failed to calculate metrics: {e}")
        import traceback

        traceback.print_exc()
        metrics = {"error": f"Metrics calculation failed: {e}"}

    output = {"results": results, "metrics": metrics}
    output = convert_numpy_types(output)
    return output


# ------------------------------------------------------------------------------
# CLASS DEFINITION
# ------------------------------------------------------------------------------


class MatbenchDiscovery:
    """Matbench Discovery tasks using Groundhog HPC."""

    REPO_URL = "https://github.com/janosh/matbench-discovery"
    REPO_REF = "main"

    @staticmethod
    def _prepare_runner_config(
        num_structures: int | "DatasetSize" | "DatasetConfig",
        repo_url: str = REPO_URL,
        repo_ref: str = REPO_REF,
    ) -> Dict[str, Any]:
        """Prepare the runner configuration based on num_structures."""
        # Need to handle DatasetSize/Config which might be passed as objects or values
        # Since we are in the script, we might not have the enums imported if they are not in this file.
        # But the user passes them.
        # If they are passed as arguments, they are serialized.
        # We need to extract value.

        # Simple heuristic: if it has 'value' attr, use it.
        subset = "full"
        seed = 42

        if hasattr(num_structures, "value"):  # Enum
            subset = num_structures.value
            # Check for seed method/attr if it's our custom Config
            if hasattr(num_structures, "seed"):
                if callable(num_structures.seed):
                    pass  # It's the method
                else:
                    seed = num_structures.seed
        elif hasattr(num_structures, "subset"):  # DatasetConfig
            subset = num_structures.subset.value
            seed = num_structures.seed
        elif isinstance(num_structures, int):
            subset = "full"
            # We handle int as limit in load_dataset
            return {
                "repo_url": repo_url,
                "repo_ref": repo_ref,
                "num_structures": num_structures,
                "dataset_subset": "full",
            }

        return {
            "repo_url": repo_url,
            "repo_ref": repo_ref,
            "dataset_subset": subset,
            "dataset_seed": seed,
        }

    @staticmethod
    def _generate_checkpoint_name(
        model_packages: str | List[str], runner_config: Dict[str, Any]
    ) -> str:
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

    @staticmethod
    def _run_task(
        model_factory: Any,
        model_packages: str | List[str],
        num_structures: int | "DatasetSize" | "DatasetConfig",
        checkpoint_name: str | None,
        checkpoint_path: str | None,
        process_fn: Any,
        load_dataset_fn: Any,
        calc_metrics_fn: Any,
        sys_path: List[str] | None = None,
    ) -> Dict[str, Any]:
        # Add custom sys.path if provided (useful for local execution/testing)
        if sys_path:
            import sys

            for p in sys_path:
                if p not in sys.path:
                    sys.path.append(p)

        runner_config = MatbenchDiscovery._prepare_runner_config(num_structures)

        if not checkpoint_name and not checkpoint_path:
            checkpoint_name = MatbenchDiscovery._generate_checkpoint_name(
                model_packages, runner_config
            )

        if checkpoint_path:
            print(f"Resuming from checkpoint: {checkpoint_path}")
            final_checkpoint_path = checkpoint_path
        else:
            print(
                f"Checkpoint will be saved to: ~/.garden/benchmarks/{checkpoint_name}"
            )
            final_checkpoint_path = os.path.expanduser(
                f"~/.garden/benchmarks/{checkpoint_name}"
            )
            # Ensure directory exists
            os.makedirs(os.path.dirname(final_checkpoint_path), exist_ok=True)

        runner_config["checkpoint_path"] = final_checkpoint_path

        return run_benchmark_hog(
            runner_config,
            model_factory,
            load_dataset_fn,
            process_fn,
            calc_metrics_fn,
        )

    @hog.method()
    def IS2RE(
        model_factory: Any,
        model_packages: str | List[str],
        num_structures: int | "DatasetSize" | "DatasetConfig" = 100,
        checkpoint_name: str | None = None,
        checkpoint_path: str | None = None,
        sys_path: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Initial Structure to Relaxed Energy."""
        return MatbenchDiscovery._run_task(
            model_factory,
            model_packages,
            num_structures,
            checkpoint_name,
            checkpoint_path,
            process_batch_relaxation,
            load_dataset_wbm_initial,
            calculate_metrics_energy,
            sys_path=sys_path,
        )

    @hog.method()
    def RS2RE(
        model_factory: Any,
        model_packages: str | List[str],
        num_structures: int | "DatasetSize" | "DatasetConfig" = 100,
        checkpoint_name: str | None = None,
        checkpoint_path: str | None = None,
        sys_path: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Relaxed Structure to Relaxed Energy."""
        return MatbenchDiscovery._run_task(
            model_factory,
            model_packages,
            num_structures,
            checkpoint_name,
            checkpoint_path,
            process_batch_static,
            load_dataset_wbm_relaxed,
            calculate_metrics_energy,
            sys_path=sys_path,
        )

    @hog.method()
    def S2EFS(
        model_factory: Any,
        model_packages: str | List[str],
        num_structures: int | "DatasetSize" | "DatasetConfig" = 100,
        checkpoint_name: str | None = None,
        checkpoint_path: str | None = None,
        sys_path: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Structure to Energy, Forces, Stress."""
        return MatbenchDiscovery._run_task(
            model_factory,
            model_packages,
            num_structures,
            checkpoint_name,
            checkpoint_path,
            process_batch_forces,
            load_dataset_mp_trj,
            calculate_metrics_forces,
            sys_path=sys_path,
        )

    # Aliases
    @hog.method()
    def S2EF(*args, **kwargs):
        return MatbenchDiscovery.S2EFS(*args, **kwargs)

    @hog.method()
    def S2EFSM(*args, **kwargs):
        return MatbenchDiscovery.S2EFS(*args, **kwargs)

    @hog.method()
    def IS2E(*args, **kwargs):
        # Same as IS2RE but static? No, IS2E is Initial Structure to Energy (Static).
        # IS2RE is Relaxation.
        # IS2E logic:
        return MatbenchDiscovery._run_task(
            *args,
            **kwargs,
            process_fn=process_batch_static,
            load_dataset_fn=load_dataset_wbm_initial,
            calc_metrics_fn=calculate_metrics_energy,
        )

    @hog.method()
    def S2E(*args, **kwargs):
        # Structure to Energy (Relaxed Structure to Energy) -> RS2RE
        return MatbenchDiscovery.RS2RE(*args, **kwargs)

    @hog.method()
    def S2RE(*args, **kwargs):
        # Structure to Relaxed Energy -> IS2RE
        return MatbenchDiscovery.IS2RE(*args, **kwargs)

    @hog.method()
    def RP2RE(*args, **kwargs):
        return MatbenchDiscovery.IS2RE(*args, **kwargs)

    @hog.method()
    def IP2E(*args, **kwargs):
        return MatbenchDiscovery.IS2E(*args, **kwargs)
