"""Test Matbench Discovery benchmark on Anvil HPC.

This script demonstrates running the IS2RE benchmark with a subset of structures
using multi-GPU parallelization on a Globus Compute endpoint.
"""

from garden_ai.benchmarks.matbench_discovery import DatasetSize, MatbenchDiscovery

# Globus Compute endpoint
ENDPOINT_ID = "5aafb4c1-27b2-40d8-a038-a0277611868f"

# HPC endpoint configuration
ENDPOINT_CONFIG = {
    "account": "cis250461-gpu",
    "partition": "gpu-debug",
    "qos": "gpu",
    "scheduler_options": "#SBATCH --gpus-per-node=2\n#SBATCH --cpus-per-task=8",
}


# Model factory function for MACE
def create_mace_model(device):
    from mace.calculators import mace_mp

    return mace_mp(model="medium", device=device, default_dtype="float64")


NUM_STRUCTURES = DatasetSize.RANDOM_100


def main():
    """Run Matbench Discovery IS2RE benchmark with MACE."""

    with MatbenchDiscovery(
        endpoint_id=ENDPOINT_ID, user_endpoint_config=ENDPOINT_CONFIG
    ) as bench:
        # Run IS2RE task (Initial Structure to Relaxed Energy)
        future = bench.tasks.IS2RE.submit(
            model_factory=create_mace_model,
            model_packages="mace-torch",
            num_structures=NUM_STRUCTURES,
        )

        print("Job submitted! Waiting for results (this may take a while)...")

        try:
            output = future.result()
            metrics = output.get("metrics", {})

            if "error" in metrics:
                print(f"error               : {metrics['error']}")
            else:
                # Discovery metrics (stability classification)
                if "F1" in metrics:
                    print(f"F1                  : {metrics['F1']:.6f}")
                    print(f"DAF                 : {metrics['DAF']:.2f}x")
                    print(f"Precision           : {metrics['Precision']:.6f}")
                    print(f"Recall              : {metrics['Recall']:.6f}")
                    print(f"Accuracy            : {metrics['Accuracy']:.6f}")

                # Regression metrics
                if "MAE" in metrics:
                    print(f"MAE (eV/atom)       : {metrics['MAE']:.6f}")
                    print(f"RMSE (eV/atom)      : {metrics['RMSE']:.6f}")
                    print(f"R2                  : {metrics['R2']:.6f}")

                # Force metrics (if S2EFS task)
                if "force_mae" in metrics:
                    print(f"force_mae           : {metrics['force_mae']:.6f}")
                    print(f"force_rmse          : {metrics['force_rmse']:.6f}")
                    print(f"force_r2            : {metrics['force_r2']:.6f}")
                    print(f"stress_mae          : {metrics['stress_mae']:.6f}")
                    print(f"stress_rmse         : {metrics['stress_rmse']:.6f}")
                    print(f"stress_r2           : {metrics['stress_r2']:.6f}")

                if "num_evaluated" in metrics:
                    print(f"num_evaluated       : {metrics['num_evaluated']}")

        except Exception as e:
            print(f"\n[ERROR] Benchmark failed: {e}")
            raise


if __name__ == "__main__":
    main()
