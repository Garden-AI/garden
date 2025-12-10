#!/usr/bin/env python3
"""Matbench Discovery Benchmark - EquiformerV2 Example"""

from garden_ai.benchmarks.matbench_discovery import MatbenchDiscovery

# Globus Compute endpoint
ENDPOINT_ID = "5aafb4c1-27b2-40d8-a038-a0277611868f"


def create_equiformerv2_model(device):
    from fairchem.core.calculate.ase_calculator import Calculator

    # Use pre-trained checkpoint - will auto-download from HuggingFace
    return Calculator(
        model_name="EquiformerV2-31M-S2EF-OC20-All+MD", cpu=(device == "cpu")
    )


def main():
    print(f"Running EquiformerV2 benchmark on endpoint {ENDPOINT_ID}...")

    # Run S2EFS task (structure to energy/forces/stress)
    output = MatbenchDiscovery.S2EFS.remote(
        endpoint=ENDPOINT_ID,
        user_endpoint_config={
            "scheduler_options": "#SBATCH --gpus-per-node=2\n#SBATCH --cpus-per-task=8\n",
            "walltime": 7200,
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

    if "error" in output.get("metrics", {}):
        print(f"Error: {output['metrics']['error']}")
    else:
        print("Benchmark Results:", output.get("metrics"))


if __name__ == "__main__":
    main()
