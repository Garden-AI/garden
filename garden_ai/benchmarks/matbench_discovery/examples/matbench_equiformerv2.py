#!/usr/bin/env python3
"""
Matbench Discovery Benchmark - EquiformerV2 Example

EquiformerV2 is an improved equivariant transformer from FAIR-Chem (formerly OCP).
Paper: https://arxiv.org/abs/2306.12059
GitHub: https://github.com/Open-Catalyst-Project/ocp

Note: This example uses the S2EFS task (Structure to Energy, Forces, Stress)
instead of IS2RE because EquiformerV2 doesn't support geometry relaxation.
"""

from garden_ai.benchmarks.matbench_discovery import MatbenchDiscovery

# =============================================================================
# Configuration
# =============================================================================

# Globus Compute endpoint
ENDPOINT_ID = "5aafb4c1-27b2-40d8-a038-a0277611868f"

# HPC endpoint configuration
ENDPOINT_CONFIG = {
    "account": "cis250461-gpu",
    "partition": "gpu-debug",
    "qos": "gpu",
    "scheduler_options": "#SBATCH --gpus-per-node=2\n#SBATCH --cpus-per-task=8",
}


# Model factory function for EquiformerV2
def create_equiformerv2_model(device):
    """Create EquiformerV2 model calculator.

    Args:
        device: Device to load model on ("cuda" or "cpu")

    Returns:
        ASE calculator for EquiformerV2
    """
    from fairchem.core.calculate.ase_calculator import Calculator

    # Use pre-trained checkpoint - will auto-download from HuggingFace
    return Calculator(
        model_name="EquiformerV2-31M-S2EF-OC20-All+MD", cpu=(device == "cpu")
    )


# Benchmark parameters
NUM_STRUCTURES = 1000
USE_MULTI_GPU = True

# =============================================================================
# Run Benchmark
# =============================================================================


def main():
    """Run Matbench Discovery S2EFS benchmark with EquiformerV2."""

    print("=" * 80)
    print("Matbench Discovery S2EFS Benchmark")
    print("=" * 80)
    print(f"Endpoint:   {ENDPOINT_ID}")
    print("Model:      EquiformerV2-31M")
    print("Task:       S2EFS (Structure to Energy, Forces, Stress)")
    print(f"Structures: {NUM_STRUCTURES}")
    print(f"Resources:  {'Multi-GPU' if USE_MULTI_GPU else 'Single GPU'}")
    print("=" * 80)
    print()

    with MatbenchDiscovery(
        endpoint_id=ENDPOINT_ID, user_endpoint_config=ENDPOINT_CONFIG
    ) as bench:
        # Run S2EFS task (uses relaxed structures, no geometry optimization)
        # This is suitable for EquiformerV2 which doesn't support relaxation
        print("Submitting S2EFS task...")
        future = bench.tasks.S2EFS.submit(
            model_factory=create_equiformerv2_model,
            model_package="fairchem-core",
            num_structures=NUM_STRUCTURES,
            use_multi_gpu=USE_MULTI_GPU,
        )

        print("Waiting for results (this may take a while)...")
        output = future.result()

        # Display metrics
        print()
        print("=" * 80)
        print("Benchmark Results")
        print("=" * 80)

        metrics = output.get("metrics", {})
        if "error" in metrics:
            print(f"Error: {metrics['error']}")
        else:
            # Energy metrics
            if "energy_mae" in metrics:
                print("Energy Metrics:")
                print(f"  MAE (eV/atom):  {metrics.get('energy_mae', 'N/A'):.6f}")
                print(f"  RMSE (eV/atom): {metrics.get('energy_rmse', 'N/A'):.6f}")
                print(f"  R²:             {metrics.get('energy_r2', 'N/A'):.6f}")
                print()

            # Force metrics
            if "force_mae" in metrics:
                print("Force Metrics:")
                print(f"  MAE (eV/Å):     {metrics.get('force_mae', 'N/A'):.6f}")
                print(f"  RMSE (eV/Å):    {metrics.get('force_rmse', 'N/A'):.6f}")
                print(f"  R²:             {metrics.get('force_r2', 'N/A'):.6f}")
                print()

            # Stress metrics
            if "stress_mae" in metrics:
                print("Stress Metrics:")
                print(f"  MAE (GPa):      {metrics.get('stress_mae', 'N/A'):.6f}")
                print(f"  RMSE (GPa):     {metrics.get('stress_rmse', 'N/A'):.6f}")
                print(f"  R²:             {metrics.get('stress_r2', 'N/A'):.6f}")
                print()

            if "num_evaluated" in metrics:
                print(f"Structures:     {metrics.get('num_evaluated', 'N/A')}")

        print("=" * 80)


if __name__ == "__main__":
    main()
