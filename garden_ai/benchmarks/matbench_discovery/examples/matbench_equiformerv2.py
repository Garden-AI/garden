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


# =============================================================================
# Model Factory
# =============================================================================


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


# =============================================================================
# Run Benchmark
# =============================================================================


def main():
    """Run Matbench Discovery S2EFS benchmark with EquiformerV2."""

    print("=" * 80)
    print("Matbench Discovery S2EFS Benchmark - EquiformerV2")
    print("=" * 80)

    # Run S2EFS task using the new groundhog API
    # S2EFS is suitable for EquiformerV2 which doesn't support relaxation
    output = MatbenchDiscovery.S2EFS.remote(
        endpoint=ENDPOINT_ID,
        user_endpoint_config={
            "scheduler_options": "#SBATCH --gpus-per-node=2\n#SBATCH --cpus-per-task=8\n",
            "walltime": 7200,  # 2 hours in seconds
            "qos": "gpu",
            "partition": "gpu-debug",
            "account": "cis250461-gpu",
            "cores_per_node": 16,
            "mem_per_node": 32,
            "requirements": "",
        },
        model_factory=create_equiformerv2_model,
        model_packages="fairchem-core",
        num_structures="random_100",
    )

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
