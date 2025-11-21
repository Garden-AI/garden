"""Remote execution functions for Matbench Discovery benchmarks.

These functions are serialized and executed on Globus Compute endpoints.
They handle environment setup, dependency installation, and benchmark execution.
"""


def run_matbench_is2re(
    repo_url: str,
    repo_ref: str,
    model_package: str,
    model_factory: str,
    model_kwargs: dict,
    model_checkpoint: str | None,
    num_structures: int,
    use_multi_gpu: bool = True,
) -> dict:
    """Run Matbench IS2RE benchmark on remote Globus Compute endpoint.

    This function performs the complete benchmark workflow:
    1. Set up Python environment with UV
    2. Install dependencies (matbench-discovery + model package)
    3. Execute benchmark runner script in the environment
    4. Return results

    Args:
        repo_url: GitHub URL for matbench-discovery repo
        repo_ref: Git branch/tag/commit to checkout
        model_package: Python package name to install (e.g., "mace-torch")
        model_factory: Function or class name to create model (e.g., "mace_mp", "MACE")
        model_kwargs: Dictionary of kwargs to pass when creating model
        model_checkpoint: Path/URL to model checkpoint file (optional)
        num_structures: Number of test structures to run (subset for MVP)
        use_multi_gpu: If True, automatically detect and use all available GPUs
                      in parallel. If False, use single GPU/CPU. (default: True)

    Returns:
        Dictionary with benchmark results:
            - energies: List of final energies (None for failed relaxations)
            - num_converged: Count of successful relaxations
            - failed_indices: List of structure indices that failed

    Raises:
        RuntimeError: If benchmark execution fails
    """
    # All imports must be inside the function for CombinedCode serialization
    import json
    import logging
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        force=True,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    logger = logging.getLogger(__name__)

    # Create isolated working directory
    work_dir = Path(tempfile.mkdtemp(prefix="matbench_benchmark_"))

    # This script runs INSIDE the virtual environment
    BENCHMARK_RUNNER_SCRIPT = '''
import json
import sys
import time
import logging
import os
import concurrent.futures
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from zipfile import ZipFile
from io import TextIOWrapper

import torch
from ase.io import read
from ase.optimize import FIRE
from matbench_discovery.data import DataFiles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [PID:%(process)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger("benchmark_runner")

def setup_device(gpu_id: Optional[int] = None) -> str:
    """Setup compute device for this process."""
    if torch.cuda.is_available():
        return f"cuda:{gpu_id}" if gpu_id is not None else "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model(config: Dict[str, Any], device: str):
    """Initialize the model from configuration."""
    package_name = config["package"]
    factory_name = config["factory"]
    kwargs = config["kwargs"].copy()
    checkpoint = config.get("checkpoint")

    if "device" in kwargs:
        kwargs["device"] = device

    # Import factory function
    module_parts = package_name.split(".")
    try:
        if len(module_parts) > 1:
            module = importlib.import_module(package_name)
        else:
            # Try common patterns for model packages
            base_module = module_parts[0].split("-")[0]
            try:
                module = importlib.import_module(f"{base_module}.calculators")
            except ImportError:
                module = importlib.import_module(base_module)

        factory = getattr(module, factory_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load model factory {factory_name} from {package_name}: {e}")

    # Create model
    model = factory(**kwargs)

    # Load checkpoint if provided
    if checkpoint and checkpoint != "None":
        if hasattr(model, "load_checkpoint"):
            model.load_checkpoint(checkpoint)
        elif hasattr(model, "load_state_dict"):
            model.load_state_dict(torch.load(checkpoint))

    return model

def process_batch(
    batch_id: int,
    structures: List[Any],
    start_idx: int,
    model_config: Dict[str, Any],
    num_threads: int
) -> Dict[str, Any]:
    """Process a batch of structures on a specific device."""

    # Configure thread limits to avoid contention
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)

    gpu_id = model_config.get("gpu_id")
    device = setup_device(gpu_id)

    worker_logger = logging.getLogger(f"worker_{batch_id}")
    worker_logger.info(f"Started on {device} with {len(structures)} structures. Threads: {num_threads}")

    try:
        model = load_model(model_config, device)
    except Exception as e:
        worker_logger.error(f"Failed to initialize model: {e}")
        return {
            "energies": [None] * len(structures),
            "num_converged": 0,
            "failed_indices": [start_idx + i for i in range(len(structures))],
            "error": str(e)
        }

    energies = []
    failed_indices = []
    num_converged = 0
    batch_start = time.time()

    for i, atoms in enumerate(structures):
        global_idx = start_idx + i
        try:
            atoms.calc = model
            opt = FIRE(atoms, logfile=None)
            opt.run(fmax=0.05, steps=500)

            energies.append(atoms.get_potential_energy())
            num_converged += 1

            if (i + 1) % 10 == 0:
                elapsed = time.time() - batch_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                worker_logger.info(f"Progress: {i+1}/{len(structures)} ({rate:.2f} struct/s)")

        except Exception as e:
            worker_logger.warning(f"Structure {global_idx} failed: {e}")
            energies.append(None)
            failed_indices.append(global_idx)

    return {
        "energies": energies,
        "num_converged": num_converged,
        "failed_indices": failed_indices
    }

def load_structures(num_structures: int) -> List[Any]:
    """Load structures from the Matbench Discovery dataset."""
    structures = []
    zip_path = DataFiles.wbm_initial_atoms.path

    with ZipFile(zip_path, 'r') as zf:
        # Sort files numerically
        file_list = sorted(
            zf.namelist(),
            key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else float('inf')
        )
        for filename in file_list[:num_structures]:
            with zf.open(filename) as f:
                text_stream = TextIOWrapper(f, encoding='utf-8')
                structures.append(read(text_stream, format='extxyz'))
    return structures

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python benchmark_runner.py <config_file>")

    with open(sys.argv[1]) as f:
        config = json.load(f)

    logger.info("Starting benchmark runner...")

    try:
        structures = load_structures(config["num_structures"])
        logger.info(f"Loaded {len(structures)} structures")
    except Exception as e:
        logger.error(f"Failed to load structures: {e}")
        sys.exit(1)

    # Shuffle for load balancing
    import random
    random.seed(42)
    random.shuffle(structures)

    # Resource detection
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_multi_gpu = config.get("use_multi_gpu", True) and num_gpus > 1

    total_cores = os.cpu_count() or 1
    num_workers = num_gpus if use_multi_gpu else 1
    # Reserve cores for overhead if possible
    available_cores = max(1, total_cores - 2) if total_cores > 4 else total_cores
    threads_per_worker = max(1, available_cores // num_workers)

    logger.info(f"Resources: {num_gpus} GPUs, {total_cores} Cores. Using {num_workers} workers ({threads_per_worker} threads/worker)")

    results = {"energies": [], "num_converged": 0, "failed_indices": []}
    start_time = time.time()

    if use_multi_gpu:
        logger.info(f"Parallel execution on {num_gpus} GPUs")
        batch_size = len(structures) // num_gpus
        futures = []

        ctx = multiprocessing.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
            for i in range(num_gpus):
                start_idx = i * batch_size
                end_idx = len(structures) if i == num_gpus - 1 else (i + 1) * batch_size

                model_config = {
                    "package": config["model_package"],
                    "factory": config["model_factory"],
                    "kwargs": config["model_kwargs"],
                    "checkpoint": config["model_checkpoint"],
                    "gpu_id": i
                }

                futures.append(executor.submit(
                    process_batch, i, structures[start_idx:end_idx], start_idx, model_config, threads_per_worker
                ))

            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_res = future.result()
                    results["energies"].extend(batch_res["energies"])
                    results["num_converged"] += batch_res["num_converged"]
                    results["failed_indices"].extend(batch_res["failed_indices"])
                except Exception as e:
                    logger.error(f"Worker failed: {e}")
    else:
        logger.info("Single process execution")
        model_config = {
            "package": config["model_package"],
            "factory": config["model_factory"],
            "kwargs": config["model_kwargs"],
            "checkpoint": config["model_checkpoint"]
        }
        results = process_batch(0, structures, 0, model_config, threads_per_worker)

    elapsed = time.time() - start_time
    logger.info(f"Benchmark complete in {elapsed:.1f}s. Converged: {results['num_converged']}/{len(structures)}")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    import multiprocessing
    main()
'''

    try:
        # ----------------------------------------------------------------------
        # 1. ENVIRONMENT SETUP
        # ----------------------------------------------------------------------
        logger.info("Step 1/4: Setting up environment...")

        # Find UV binary
        uv_bin = subprocess.check_output(
            [sys.executable, "-c", "import uv; print(uv.find_uv_bin())"], text=True
        ).strip()

        # Create UV virtual environment
        subprocess.run(
            [uv_bin, "venv", "--python", "3.11"],
            cwd=work_dir,
            check=True,
            capture_output=True,
        )

        venv_python = work_dir / ".venv/bin/python"
        if not venv_python.exists():
            venv_python = work_dir / ".venv/Scripts/python.exe"  # Windows fallback

        if not venv_python.exists():
            raise RuntimeError(f"Virtual environment python not found at {venv_python}")

        # Install dependencies
        logger.info("Installing dependencies...")
        subprocess.run(
            [
                uv_bin,
                "pip",
                "install",
                "--python",
                str(venv_python),
                "matbench-discovery",
            ],
            cwd=work_dir,
            check=True,
        )
        subprocess.run(
            [uv_bin, "pip", "install", "--python", str(venv_python), model_package],
            cwd=work_dir,
            check=True,
        )

        # Set SSL cert file for HPC
        env = dict(os.environ)
        env["MBD_AUTO_DOWNLOAD_FILES"] = "true"

        try:
            certifi_path = subprocess.check_output(
                [str(venv_python), "-c", "import certifi; print(certifi.where())"],
                text=True,
            ).strip()
            env["SSL_CERT_FILE"] = certifi_path
        except Exception as e:
            logger.warning(f"Failed to set SSL_CERT_FILE: {e}")

        # ----------------------------------------------------------------------
        # 2. PREPARE BENCHMARK SCRIPT
        # ----------------------------------------------------------------------
        logger.info("Step 2/4: Preparing benchmark script...")

        # Write runner script
        runner_path = work_dir / "benchmark_runner.py"
        runner_path.write_text(BENCHMARK_RUNNER_SCRIPT)

        # Write config
        config = {
            "repo_url": repo_url,
            "repo_ref": repo_ref,
            "model_package": model_package,
            "model_factory": model_factory,
            "model_kwargs": model_kwargs,
            "model_checkpoint": model_checkpoint,
            "num_structures": num_structures,
            "use_multi_gpu": use_multi_gpu,
        }

        config_path = work_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # ----------------------------------------------------------------------
        # 3. EXECUTE BENCHMARK
        # ----------------------------------------------------------------------
        logger.info("Step 3/4: Executing benchmark...")

        # Run the runner script inside the venv
        # We stream output directly to stdout so the user sees progress
        proc = subprocess.run(
            [str(venv_python), str(runner_path), str(config_path)],
            cwd=work_dir,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=False,  # We check return code manually
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"Benchmark runner failed with return code {proc.returncode}"
            )

        # ----------------------------------------------------------------------
        # 4. COLLECT RESULTS
        # ----------------------------------------------------------------------
        logger.info("Step 4/4: Collecting results...")

        results_path = work_dir / "results.json"
        if not results_path.exists():
            raise RuntimeError(
                "Results file not found - benchmark may have crashed silently"
            )

        with open(results_path) as f:
            results = json.load(f)

        logger.info("Benchmark completed successfully.")
        return results

    finally:
        # Cleanup working directory
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)
