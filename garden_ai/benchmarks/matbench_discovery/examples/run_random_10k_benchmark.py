#!/usr/bin/env python3
"""
Run Matbench Discovery benchmarks on 10k random structures.

This script benchmarks MACE, MatterSim, and SevenNet on a random 10k
sample from the unique prototypes subset and saves comprehensive metrics to JSON.
"""

import json
from datetime import datetime
from pathlib import Path

from garden_ai.benchmarks.matbench_discovery import MatbenchDiscovery

# =============================================================================
# Configuration
# =============================================================================

# Globus Compute endpoint
ENDPOINT_ID = "5aafb4c1-27b2-40d8-a038-a0277611868f"

# Common endpoint configuration
ENDPOINT_CONFIG = {
    "scheduler_options": "#SBATCH --gpus-per-node=4\n",
    "walltime": 14400,  # 4 hours in seconds
    "qos": "gpu",
    "partition": "gpu",
    "account": "cis250461-gpu",
    "cores_per_node": 16,
    "mem_per_node": 64,
    "requirements": "",
}

# Output file for metrics
OUTPUT_FILE = "random_10k_benchmark_results.json"


# =============================================================================
# Model Factory Functions
# =============================================================================


def create_mace_model(device):
    """Create MACE model calculator."""
    from mace.calculators import mace_mp

    return mace_mp(model="medium-mpa-0", device=device, default_dtype="float64")


def create_mattersim_model(device):
    """Create MatterSim model calculator."""
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(device=device)


def create_sevennet_model(device):
    """Create SevenNet model calculator."""
    from sevenn.calculator import SevenNetCalculator

    return SevenNetCalculator(model="7net-l3i5", device=device)


# Model configurations
MODELS = {
    "MACE": {
        "packages": [
            "mace-torch",
            "cuequivariance",
            "cuequivariance-torch",
            "cuequivariance-ops-torch-cu12",
        ],
        "factory": create_mace_model,
    },
    "MatterSim": {
        "packages": ["mattersim"],
        "factory": create_mattersim_model,
    },
    "SevenNet": {
        "packages": ["sevenn"],
        "factory": create_sevennet_model,
    },
}


# =============================================================================
# Run Benchmarks
# =============================================================================


def main():
    """Run benchmarks on all models and save results."""

    print("=" * 80)
    print("Matbench Discovery Benchmark - Random 10k")
    print("=" * 80)
    print("Dataset: Random 10k from Unique Prototypes")
    print(f"Models: {', '.join(MODELS.keys())}")
    print(f"Endpoint: {ENDPOINT_ID}")
    print("=" * 80)
    print()

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset": "random_10k",
            "dataset_size": 10000,
            "endpoint_id": ENDPOINT_ID,
        },
        "models": {},
    }

    for model_name, config in MODELS.items():
        print(f"\n{'=' * 80}")
        print(f"Running {model_name}...")
        print(f"{'=' * 80}\n")

        try:
            # Run benchmark using the new groundhog API
            output = MatbenchDiscovery.IS2RE.remote(
                endpoint=ENDPOINT_ID,
                user_endpoint_config=ENDPOINT_CONFIG,
                model_factory=config["factory"],
                model_packages=config["packages"],
                num_structures="random_10k",
            )

            # Store complete output (contains both metrics and per-structure results)
            results["models"][model_name] = {
                "status": "success",
                **output,
            }

            # Display metrics
            metrics = output.get("metrics", {})
            if "error" in metrics:
                print(f"❌ {model_name} failed: {metrics['error']}")
                results["models"][model_name]["status"] = "failed"
                results["models"][model_name]["error"] = metrics["error"]
            else:
                print(f"✅ {model_name} completed successfully!")
                print(f"   F1 Score:       {metrics.get('F1', 'N/A'):.6f}")
                print(f"   DAF:            {metrics.get('DAF', 'N/A'):.2f}x")
                print(f"   MAE (eV/atom):  {metrics.get('MAE', 'N/A'):.6f}")
                print(f"   RMSE (eV/atom): {metrics.get('RMSE', 'N/A'):.6f}")
                print(f"   Structures:     {metrics.get('num_evaluated', 'N/A')}")

        except Exception as e:
            print(f"❌ {model_name} error: {e}")
            results["models"][model_name] = {
                "status": "error",
                "error": str(e),
            }

    # Save results to JSON
    output_path = Path(OUTPUT_FILE)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print("Benchmark Complete!")
    print(f"{'=' * 80}")
    print(f"\nResults saved to: {output_path.absolute()}")

    # Print summary table
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}\n")
    print(f"{'Model':<15} {'Status':<10} {'F1':<10} {'DAF':<10} {'MAE':<10}")
    print("-" * 80)

    for model_name, data in results["models"].items():
        if data["status"] == "success":
            metrics = data["metrics"]
            print(
                f"{model_name:<15} {data['status']:<10} "
                f"{metrics.get('F1', 0):<10.6f} "
                f"{metrics.get('DAF', 0):<10.2f} "
                f"{metrics.get('MAE', 0):<10.6f}"
            )
        else:
            print(
                f"{model_name:<15} {data['status']:<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}"
            )

    print()


if __name__ == "__main__":
    main()
