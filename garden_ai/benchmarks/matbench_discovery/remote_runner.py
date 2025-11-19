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
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    # Ensure stdout is unbuffered
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
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] [PID:%(process)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger("benchmark_runner")

def setup_device(gpu_id: Optional[int] = None) -> str:
    """Setup compute device for this process."""
    import torch

    if gpu_id is not None and torch.cuda.is_available():
        # Set visible devices to just this GPU to avoid contention
        # and ensure model uses the correct device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return "cuda:0"
    elif torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def process_batch(
    batch_id: int,
    structures: List[Any],
    start_idx: int,
    model_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Process a batch of structures on a specific device."""

    # Setup logging for this worker
    worker_logger = logging.getLogger(f"worker_{batch_id}")
    worker_logger.setLevel(logging.INFO)

    gpu_id = model_config.get("gpu_id")
    device = setup_device(gpu_id)
    worker_logger.info(f"Worker {batch_id} started on {device} with {len(structures)} structures")

    # Initialize model
    try:
        import importlib

        package_name = model_config["package"]
        factory_name = model_config["factory"]
        kwargs = model_config["kwargs"].copy()
        checkpoint = model_config.get("checkpoint")

        # Update device in kwargs
        if "device" in kwargs:
            kwargs["device"] = device

        # Import factory
        module_parts = package_name.split(".")
        if len(module_parts) > 1:
            module = importlib.import_module(package_name)
            factory = getattr(module, factory_name)
        else:
            base_module = module_parts[0].split("-")[0]
            try:
                module = importlib.import_module(f"{base_module}.calculators")
                factory = getattr(module, factory_name)
            except (ImportError, AttributeError):
                module = importlib.import_module(base_module)
                factory = getattr(module, factory_name)

        # Create model
        model = factory(**kwargs)

        # Load checkpoint
        if checkpoint and checkpoint != "None":
            if hasattr(model, "load_checkpoint"):
                model.load_checkpoint(checkpoint)
            elif hasattr(model, "load_state_dict"):
                import torch
                model.load_state_dict(torch.load(checkpoint))

    except Exception as e:
        worker_logger.error(f"Failed to initialize model: {e}")
        return {
            "energies": [None] * len(structures),
            "num_converged": 0,
            "failed_indices": [start_idx + i for i in range(len(structures))],
            "error": str(e)
        }

    # Run relaxations
    from ase.optimize import FIRE

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

            energy = atoms.get_potential_energy()
            energies.append(energy)
            num_converged += 1

            # Log progress occasionally
            if (i + 1) % 10 == 0:
                elapsed = time.time() - batch_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(structures) - i - 1) / rate if rate > 0 else 0
                worker_logger.info(
                    f"Progress: {i+1}/{len(structures)} "
                    f"({rate:.2f} struct/s, ETA: {eta/60:.1f}m)"
                )

        except Exception as e:
            worker_logger.warning(f"Structure {global_idx} failed: {e}")
            energies.append(None)
            failed_indices.append(global_idx)

    return {
        "energies": energies,
        "num_converged": num_converged,
        "failed_indices": failed_indices
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python benchmark_runner.py <config_file>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path) as f:
        config = json.load(f)

    logger.info("Starting benchmark runner...")

    # Load structures
    logger.info("Loading structures...")
    try:
        from matbench_discovery.data import DataFiles
        from zipfile import ZipFile
        from ase.io import read
        from io import TextIOWrapper

        structures = []
        zip_path = DataFiles.wbm_initial_atoms.path
        num_structures = config["num_structures"]

        with ZipFile(zip_path, 'r') as zf:
            file_list = sorted(
                zf.namelist(),
                key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else float('inf')
            )
            for i, filename in enumerate(file_list[:num_structures]):
                with zf.open(filename) as f:
                    text_stream = TextIOWrapper(f, encoding='utf-8')
                    atoms = read(text_stream, format='extxyz')
                    structures.append(atoms)

        logger.info(f"Loaded {len(structures)} structures")

    except Exception as e:
        logger.error(f"Failed to load structures: {e}")
        sys.exit(1)

    # Determine parallelization strategy
    import torch
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_multi_gpu = config.get("use_multi_gpu", True) and num_gpus > 1

    results = {
        "energies": [],
        "num_converged": 0,
        "failed_indices": []
    }

    start_time = time.time()

    if use_multi_gpu:
        logger.info(f"Running on {num_gpus} GPUs in parallel")

        # Split structures
        batch_size = len(structures) // num_gpus
        futures = []

        # Use 'spawn' start method for CUDA compatibility
        import multiprocessing
        ctx = multiprocessing.get_context('spawn')

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
            for i in range(num_gpus):
                start_idx = i * batch_size
                end_idx = len(structures) if i == num_gpus - 1 else (i + 1) * batch_size
                batch_structures = structures[start_idx:end_idx]

                model_config = {
                    "package": config["model_package"],
                    "factory": config["model_factory"],
                    "kwargs": config["model_kwargs"],
                    "checkpoint": config["model_checkpoint"],
                    "gpu_id": i
                }

                futures.append(
                    executor.submit(
                        process_batch,
                        i,
                        batch_structures,
                        start_idx,
                        model_config
                    )
                )

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_res = future.result()
                    results["energies"].extend(batch_res["energies"])
                    results["num_converged"] += batch_res["num_converged"]
                    results["failed_indices"].extend(batch_res["failed_indices"])
                except Exception as e:
                    logger.error(f"Worker failed: {e}")

    else:
        logger.info("Running in single process")
        model_config = {
            "package": config["model_package"],
            "factory": config["model_factory"],
            "kwargs": config["model_kwargs"],
            "checkpoint": config["model_checkpoint"],
            # No gpu_id means let model decide or use default
        }

        batch_res = process_batch(0, structures, 0, model_config)
        results = batch_res

    elapsed = time.time() - start_time
    logger.info(f"Benchmark complete in {elapsed:.1f}s")
    logger.info(f"Converged: {results['num_converged']}/{len(structures)}")

    # Save results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
'''

    try:
        # ----------------------------------------------------------------------
        # 1. ENVIRONMENT SETUP
        # ----------------------------------------------------------------------
        logger.info("Step 1/4: Setting up environment...")

        uv_bin = (
            subprocess.run(
                ["python", "-c", "import uv; print(uv.find_uv_bin())"],
                capture_output=True,
            )
            .stdout.decode("utf-8")
            .strip()
        )

        # Create UV virtual environment
        subprocess.run(
            [uv_bin, "venv", "--python", "3.11"],
            cwd=work_dir,
            check=True,
            capture_output=True,
            text=True,
        )

        venv_python = work_dir / ".venv/bin/python"
        if not venv_python.exists():
            # Windows path
            venv_python = work_dir / ".venv/Scripts/python.exe"

        if not venv_python.exists():
            raise RuntimeError(f"Virtual environment python not found at {venv_python}")

        # Install matbench-discovery and model package
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

        # Set SSL cert file to certifi's CA bundle to fix HPC SSL verification issues
        env = dict(os.environ)
        env["MBD_AUTO_DOWNLOAD_FILES"] = "true"

        try:
            certifi_path = subprocess.run(
                [str(venv_python), "-c", "import certifi; print(certifi.where())"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
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
