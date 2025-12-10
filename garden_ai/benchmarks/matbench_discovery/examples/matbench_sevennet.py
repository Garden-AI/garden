#!/usr/bin/env python3
"""Matbench Discovery Benchmark - SevenNet Example"""

from garden_ai.benchmarks.matbench_discovery import MatbenchDiscovery

# Globus Compute endpoint (replace with your endpoint UUID)
ENDPOINT_ID = "5aafb4c1-27b2-40d8-a038-a0277611868f"


def create_sevennet_model(device):
    from sevenn.calculator import SevenNetCalculator

    return SevenNetCalculator(model="7net-0", device=device)


def main():
    print(f"Running SevenNet benchmark on endpoint {ENDPOINT_ID}...")

    # Run IS2RE task
    output = MatbenchDiscovery.IS2RE.remote(
        endpoint=ENDPOINT_ID,
        user_endpoint_config={
            "scheduler_options": "#SBATCH --gpus-per-node=2\n",
            "walltime": 7200,
            "qos": "gpu",
            "partition": "gpu-debug",
            "account": "cis250461-gpu",
            "cores_per_node": 16,
            "mem_per_node": 32,
            "requirements": "",
        },
        model_factory=create_sevennet_model,
        model_packages="sevenn",
        num_structures="random_100",
    )

    if "error" in output.get("metrics", {}):
        print(f"Error: {output['metrics']['error']}")
    else:
        print("Benchmark Results:", output.get("metrics"))


if __name__ == "__main__":
    main()
