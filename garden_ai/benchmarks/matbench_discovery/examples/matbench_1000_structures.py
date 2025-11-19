"""Test Matbench Discovery benchmark on Anvil HPC with 1000 structures.

This script demonstrates scaling to 1000 structures using 4 GPUs in parallel.
It's designed to test the multi-GPU parallelization implementation and measure
throughput before attempting the full dataset.
"""

from garden_ai.benchmarks import MatbenchDiscovery

ENDPOINT_ID = "5aafb4c1-27b2-40d8-a038-a0277611868f"

ENDPOINT_CONFIG = {
    "account": "cis250461-gpu",  # HPC allocation/account
    "qos": "gpu",
    "partition": "gpu-debug",  # Use full partition (not debug) for longer run
    "scheduler_options": "#SBATCH --gpus-per-node=2\n#SBATCH --time=00:30:00\n#SBATCH --mem=32G",
    "worker_init": "pip install --user uv",  # Install uv on worker startup
}

MODEL_PACKAGE = "mace-torch"
MODEL_FACTORY = "mace_mp"
MODEL_KWARGS = {
    "model": "medium",
    "device": "cuda",  # Use GPU on HPC
    "default_dtype": "float64",
}

NUM_STRUCTURES = 1000

print("=" * 80)
print("Matbench Discovery IS2RE Benchmark - 1000 Structures")
print("=" * 80)
print(f"Endpoint: {ENDPOINT_ID}")
print(f"Model: {MODEL_PACKAGE} / {MODEL_FACTORY}")
print(f"Structures: {NUM_STRUCTURES}")
print("Multi-GPU: Enabled (2 GPUs)")
print("=" * 80)

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

    print("\nJob submitted! Waiting for results...")
    print("This may take a while. You can monitor progress in the Globus Compute logs.")
    print()

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

        # Calculate and display throughput
        if "num_converged" in result and result["num_converged"] > 0:
            print("\nPerformance:")
            print(f"  Success rate: {metrics.get('success_rate', 0):.1%}")
            print("  Note: Check job logs for detailed throughput (structures/hour)")

    except Exception as e:
        print(f"\n[ERROR] Benchmark failed: {e}")
        raise
