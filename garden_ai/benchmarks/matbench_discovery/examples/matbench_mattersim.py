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

# HPC endpoint configuration
ENDPOINT_CONFIG = {
    "account": "cis250461-gpu",
    "partition": "gpu-debug",
    "qos": "gpu",
    "scheduler_options": "#SBATCH --gpus-per-node=2\n#SBATCH --cpus-per-task=8",
}


# Model factory function for MatterSim
def create_mattersim_model(device):
    """Create MatterSim model calculator.

    Args:
        device: Device to load model on ("cuda" or "cpu")

    Returns:
        ASE calculator for MatterSim
    """
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(device=device)


# Benchmark parameters
NUM_STRUCTURES = 1000
USE_MULTI_GPU = True

# =============================================================================
# Run Benchmark
# =============================================================================


def main():
    """Run Matbench Discovery IS2RE benchmark with MatterSim."""

    print("=" * 80)
    print("Matbench Discovery IS2RE Benchmark")
    print("=" * 80)
    print(f"Endpoint:   {ENDPOINT_ID}")
    print("Model:      MatterSim")
    print(f"Structures: {NUM_STRUCTURES}")
    print(f"Resources:  {'Multi-GPU' if USE_MULTI_GPU else 'Single GPU'}")
    print("=" * 80)
    print()

    with MatbenchDiscovery(
        endpoint_id=ENDPOINT_ID, user_endpoint_config=ENDPOINT_CONFIG
    ) as bench:
        # Run IS2RE task
        print("Submitting IS2RE task...")
        future = bench.tasks.IS2RE.submit(
            model_factory=create_mattersim_model,
            model_package="mattersim",
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
