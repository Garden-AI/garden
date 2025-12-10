#!/usr/bin/env python3
"""Matbench Discovery Benchmark - Local Execution Example"""

from garden_ai.benchmarks.matbench_discovery import MatbenchDiscovery


def create_mattersim_model(device):
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(device=device)


def main():
    print("Running MatterSim benchmark locally...")

    # Run IS2RE task locally
    # Note: Requires a GPU or MPS if using MatterSim, or CPU if specified/supported
    output = MatbenchDiscovery.IS2RE.local(
        model_factory=create_mattersim_model,
        model_packages="mattersim",
        num_structures="random_100",
    )

    if "error" in output.get("metrics", {}):
        print(f"Error: {output['metrics']['error']}")
    else:
        print("Benchmark Results:", output.get("metrics"))


if __name__ == "__main__":
    main()
