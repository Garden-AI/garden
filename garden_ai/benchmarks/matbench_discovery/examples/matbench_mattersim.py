#!/usr/bin/env python3
"""Matbench Discovery Benchmark - MatterSim Example"""

from garden_ai.benchmarks.matbench_discovery import MatbenchDiscovery


def create_mattersim_model(device):
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(device=device)


output = MatbenchDiscovery.IS2RE.remote(
    endpoint="anvil",
    account="your-account-here",
    model_factory=create_mattersim_model,
    model_packages="mattersim",
    num_structures="random_100",
)

if "error" in output.get("metrics", {}):
    print(f"Error: {output['metrics']['error']}")
else:
    print("Benchmark Results:", output)
