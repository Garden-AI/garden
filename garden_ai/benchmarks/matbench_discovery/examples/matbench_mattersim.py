#!/usr/bin/env python3
"""
Matbench Discovery Benchmark - MatterSim Example

MatterSim is a deep learning atomistic model for general material simulations.
Paper: https://arxiv.org/abs/2405.04967
GitHub: https://github.com/microsoft/mattersim
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


def create_mattersim_model(device):
    """Create MatterSim model calculator.

    Args:
        device: Device to load model on ("cuda" or "cpu")

    Returns:
        ASE calculator for MatterSim
    """
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(device=device)


# =============================================================================
# Run Benchmark
# =============================================================================


def main():
    """Run Matbench Discovery IS2RE benchmark with MatterSim."""

    print("=" * 80)
    print("Matbench Discovery IS2RE Benchmark - MatterSim")
    print("=" * 80)

    # Run IS2RE task using the new groundhog API
    output = MatbenchDiscovery.IS2RE.remote(
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
        model_factory=create_mattersim_model,
        model_packages="mattersim",
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
        # Discovery metrics
        print(f"F1 Score:       {metrics.get('F1', 'N/A'):.6f}")
        print(f"DAF:            {metrics.get('DAF', 'N/A'):.2f}x")
        print(f"Precision:      {metrics.get('Precision', 'N/A'):.6f}")
        print(f"Recall:         {metrics.get('Recall', 'N/A'):.6f}")
        print(f"Accuracy:       {metrics.get('Accuracy', 'N/A'):.6f}")
        print()
        # Regression metrics
        print(f"MAE (eV/atom):  {metrics.get('MAE', 'N/A'):.6f}")
        print(f"RMSE (eV/atom): {metrics.get('RMSE', 'N/A'):.6f}")
        print(f"R²:             {metrics.get('R2', 'N/A'):.6f}")
        print()
        print(f"Structures:     {metrics.get('num_evaluated', 'N/A')}")

    print("=" * 80)


if __name__ == "__main__":
    main()
