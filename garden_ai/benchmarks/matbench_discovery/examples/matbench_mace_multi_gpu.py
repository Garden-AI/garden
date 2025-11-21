"""Test Matbench Discovery benchmark on Anvil HPC.

This script demonstrates running the IS2RE benchmark with a subset of structures
using multi-GPU parallelization on a Globus Compute endpoint.
"""

from garden_ai.benchmarks import MatbenchDiscovery

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Globus Compute Endpoint ID (Anvil HPC)
ENDPOINT_ID = "5aafb4c1-27b2-40d8-a038-a0277611868f"

# Job Configuration
NUM_GPUS = 2
ENDPOINT_CONFIG = {
    "account": "cis250461-gpu",
    "qos": "gpu",
    "partition": "gpu",
    "scheduler_options": f"#SBATCH --gpus-per-node={NUM_GPUS}\n#SBATCH --time=00:30:00\n",
    "cores_per_node": 32,
    "mem_per_node": 32,  # GB
    "worker_init": "pip install --user uv",  # Ensure uv is available
}

# Model Configuration
MODEL_PACKAGE = "mace-torch"
MODEL_FACTORY = "mace_mp"
MODEL_KWARGS = {
    "model": "medium",
    "device": "cuda",
    "default_dtype": "float64",
}

# Benchmark Configuration
NUM_STRUCTURES = 500


def main():
    print("=" * 80)
    print("Matbench Discovery IS2RE Benchmark")
    print("=" * 80)
    print(f"Endpoint:   {ENDPOINT_ID}")
    print(f"Model:      {MODEL_PACKAGE} / {MODEL_FACTORY}")
    print(f"Structures: {NUM_STRUCTURES}")
    print(f"Resources:  {NUM_GPUS} GPUs (Multi-GPU Enabled)")
    print("=" * 80)

    with MatbenchDiscovery(
        endpoint_id=ENDPOINT_ID,
        user_endpoint_config=ENDPOINT_CONFIG,
    ) as bench:
        task = bench.tasks.IS2RE

        print("\nSubmitting task to endpoint...")
        future = task.submit(
            model_package=MODEL_PACKAGE,
            model_factory=MODEL_FACTORY,
            model_kwargs=MODEL_KWARGS,
            num_structures=NUM_STRUCTURES,
            use_multi_gpu=True,
        )

        print("Job submitted! Waiting for results (this may take a while)...")

        try:
            result = future.result()
            metrics = task.calculate_metrics(result)

            print("\n" + "=" * 80)
            print("Benchmark Results")
            print("=" * 80)

            # Print primary metrics
            for key, value in metrics.items():
                print(f"{key:<20}: {value}")

            print("-" * 80)
            print(f"Converged: {result['num_converged']} / {NUM_STRUCTURES}")
            print(f"Failed:    {len(result.get('failed_indices', []))}")

            if result.get("energies"):
                valid_energies = [e for e in result["energies"] if e is not None]
                if valid_energies:
                    print(f"Sample energies:   {valid_energies[:3]} ...")

            print("=" * 80)

        except Exception as e:
            print(f"\n[ERROR] Benchmark failed: {e}")
            raise


if __name__ == "__main__":
    main()
