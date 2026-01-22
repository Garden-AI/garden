#!/usr/bin/env python3
"""Matbench Discovery Benchmark - SevenNet Example"""

from garden_ai.benchmarks.matbench_discovery import MatbenchDiscovery


def create_sevennet_model(device):
    from sevenn.calculator import SevenNetCalculator

    return SevenNetCalculator(model="7net-0", device=device)


output = MatbenchDiscovery.IS2RE.remote(
    endpoint="anvil",
    account="your-account-here",
    model_factory=create_sevennet_model,
    model_packages="sevenn",
    num_structures="random_100",
)

if "error" in output.get("metrics", {}):
    print(f"Error: {output['metrics']['error']}")
else:
    print("Benchmark Results:", output)
