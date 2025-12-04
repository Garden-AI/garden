#!/usr/bin/env python3
"""
Test Matbench Discovery refactor with Groundhog HPC.
"""

import os

from dummy_model import create_dummy_model

from garden_ai.benchmarks.matbench_discovery import MatbenchDiscovery

# =============================================================================
# Configuration
# =============================================================================

# Globus Compute endpoint (use local if possible, or the one from example)
ENDPOINT_ID = "5aafb4c1-27b2-40d8-a038-a0277611868f"

# HPC endpoint configuration
ENDPOINT_CONFIG = {
    "account": "cis250461-gpu",
    "partition": "gpu",
    "qos": "gpu",
    "scheduler_options": "#SBATCH --gpus-per-node=1\n",
    "cores_per_node": 4,
    "mem_per_node": 16,
}

# =============================================================================
# Model Factory Functions
# =============================================================================


def main():
    """Run benchmarks on all models and save results."""

    print("=" * 80)
    print("Matbench Discovery Test - Groundhog Refactor")
    print("=" * 80)

    print("Running LOCAL test...")

    # Ensure subprocess can find dummy_model
    cwd = os.getcwd()
    os.environ["PYTHONPATH"] = cwd + os.pathsep + os.environ.get("PYTHONPATH", "")

    try:
        # Run locally using the new static method API
        output = MatbenchDiscovery.IS2RE.local(
            model_factory=create_dummy_model,
            model_packages=["numpy", "ase"],  # Minimal deps
            num_structures=1,
            sys_path=[os.getcwd()],
        )
        print("Local run output keys:", output.keys())
        if "error" in output.get("metrics", {}):
            print("Local metrics error:", output["metrics"]["error"])
        else:
            print("Local run successful!")
            print("Metrics:", output.get("metrics"))

    except Exception as e:
        print(f"Local run failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
