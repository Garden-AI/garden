import concurrent.futures
import json
import logging
import multiprocessing
import os
import sys
import time
from typing import Optional

# ------------------------------------------------------------------------------
# BOILERPLATE: Logging & Device Setup
# ------------------------------------------------------------------------------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PID:%(process)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("benchmark_runner")


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
# USER DEFINED FUNCTIONS (Injected)
# ------------------------------------------------------------------------------
# - load_model(config, device)
# - process_batch(batch_id, batch_data, model_config, num_threads)
# - load_dataset(config) -> List[Any]
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# MAIN EXECUTION LOOP
# ------------------------------------------------------------------------------


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python benchmark_runner.py <config_file>")

    with open(sys.argv[1]) as f:
        config = json.load(f)

    logger.info("Starting benchmark runner...")

    checkpoint_path = config.get("checkpoint_path")
    results = {}

    # Load existing checkpoint if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            with open(checkpoint_path) as f:
                results = json.load(f)
            logger.info(f"Found {len(results)} processed items in checkpoint")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")

    # Load Dataset
    try:
        all_items = load_dataset(config)  # noqa: F821
        logger.info(f"Loaded {len(all_items)} total items")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Filter out already processed items
    # Assuming items are (id, data) tuples
    items_to_process = [
        (item_id, item) for item_id, item in all_items if str(item_id) not in results
    ]

    if not items_to_process:
        logger.info("All items already processed!")
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    logger.info(f"Processing {len(items_to_process)} remaining items")

    # Shuffle for load balancing
    import random

    random.seed(42)
    random.shuffle(items_to_process)

    # Resource detection
    try:
        import torch

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        num_gpus = 0

    use_multi_gpu = config.get("use_multi_gpu", True) and num_gpus > 1

    total_cores = os.cpu_count() or 1
    num_workers = num_gpus if use_multi_gpu else 1
    # Reserve some cores for system/overhead if possible
    available_cores = max(1, total_cores - 2) if total_cores > 4 else total_cores
    threads_per_worker = max(1, available_cores // num_workers)

    logger.info(
        f"Resources: {num_gpus} GPUs, {total_cores} Cores. Using {num_workers} workers ({threads_per_worker} threads/worker)"
    )

    start_time = time.time()

    # Chunk items into smaller batches to allow frequent checkpointing
    chunk_size = 1000 * num_workers
    chunks = [
        items_to_process[i : i + chunk_size]
        for i in range(0, len(items_to_process), chunk_size)
    ]

    logger.info(f"Split into {len(chunks)} chunks for processing")

    ctx = multiprocessing.get_context("spawn")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers, mp_context=ctx
    ) as executor:
        for chunk_idx, chunk in enumerate(chunks):
            chunk_start = time.time()
            logger.info(
                f"Starting chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} items)"
            )

            # Split chunk among workers
            futures = []
            batch_size = (len(chunk) + num_workers - 1) // num_workers

            for i in range(num_workers):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(chunk))
                if start < end:
                    batch = chunk[start:end]

                    # Inject worker specific config
                    worker_config = config.copy()
                    worker_config["gpu_id"] = i if use_multi_gpu else None

                    futures.append(
                        executor.submit(
                            process_batch,  # noqa: F821
                            i,
                            batch,
                            worker_config,
                            threads_per_worker,
                        )
                    )

            # Collect results for this chunk
            chunk_results = {}
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_res = future.result()
                    chunk_results.update(batch_res)
                except Exception as e:
                    logger.error(f"Worker failed in chunk {chunk_idx}: {e}")
                    import traceback

                    traceback.print_exc()
                    # Critical failure - abort benchmark immediately
                    logger.error("Aborting benchmark due to worker failure")
                    sys.exit(1)

            # Update main results and save checkpoint
            results.update(chunk_results)

            if checkpoint_path:
                try:
                    tmp_path = checkpoint_path + ".tmp"
                    with open(tmp_path, "w") as f:
                        # Convert numpy types before saving checkpoint
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

    # Calculate metrics from results
    logger.info("Calculating metrics...")
    try:
        metrics = calculate_metrics_remote(results, config)  # noqa: F821
        logger.info(f"Metrics calculated: {metrics}")
    except Exception as e:
        logger.error(f"Failed to calculate metrics: {e}")
        import traceback

        traceback.print_exc()
        metrics = {"error": f"Metrics calculation failed: {e}"}

    # Write both results and metrics
    output = {"results": results, "metrics": metrics}

    # Custom JSON encoder to handle numpy types
    # convert_numpy_types moved to global scope

    # Convert numpy types before serialization
    output = convert_numpy_types(output)

    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
