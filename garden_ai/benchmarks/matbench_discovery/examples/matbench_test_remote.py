"""Test Matbench Discovery benchmark on remote a HPC endpoint."""

from garden_ai.benchmarks import MatbenchDiscovery

ENDPOINT_ID = "5aafb4c1-27b2-40d8-a038-a0277611868f"

ENDPOINT_CONFIG = {
    "account": "cis250461-gpu",  # HPC allocation/account
    "qos": "gpu",
    "partition": "gpu-debug",  # SLURM partition
    "scheduler_options": "#SBATCH --gpus-per-node=4",  # Request 4 GPUs
    "worker_init": "pip install --user uv",  # Install uv on worker startup
}

MODEL_PACKAGE = "mace-torch"
MODEL_FACTORY = "mace_mp"
MODEL_KWARGS = {
    "model": "medium",
    "device": "cuda",  # Use GPU on HPC
    "default_dtype": "float32",
}

NUM_STRUCTURES = 100  # Increased from 10 to test multi-GPU parallelization

with MatbenchDiscovery(
    endpoint_id=ENDPOINT_ID,
    user_endpoint_config=ENDPOINT_CONFIG,
) as bench:
    task = bench.tasks.IS2RE

    future = task.submit(
        model_package=MODEL_PACKAGE,
        model_factory=MODEL_FACTORY,
        model_kwargs=MODEL_KWARGS,
        num_structures=NUM_STRUCTURES,
        use_multi_gpu=True,  # Enable multi-GPU parallelization
    )

    try:
        result = future.result()
        metrics = task.calculate_metrics(result)

        print("\nResults:")
        print("=" * 80)
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        print("=" * 80)
        print("\nRaw Results:")
        print(f"  Converged: {result['num_converged']}")
        print(f"  Failed: {len(result.get('failed_indices', []))}")
        if result.get("energies"):
            valid_energies = [e for e in result["energies"] if e is not None]
            if valid_energies:
                print(f"  Sample energies: {valid_energies[:3]}")

    except Exception as e:
        print(f"\n[ERROR] Benchmark failed: {e}")
        raise
