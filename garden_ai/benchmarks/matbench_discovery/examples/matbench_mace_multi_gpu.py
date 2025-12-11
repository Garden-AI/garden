#!/usr/bin/env python3
"""Matbench Discovery Benchmark - MACE Multi-GPU Example"""

from rich import print

from garden_ai.benchmarks.matbench_discovery import MatbenchDiscovery


def create_mace_model(device):
    from mace.calculators import mace_mp

    return mace_mp(model="medium-mpa-0", device=device, default_dtype="float64")


print("Running MACE benchmark on endpoint anvil...")

results = MatbenchDiscovery.IS2RE.remote(
    endpoint="anvil",
    account="cis250461-gpu",
    model_factory=create_mace_model,
    model_packages=[
        "mace-torch",
        "cuequivariance",
        "cuequivariance-torch",
        "cuequivariance-ops-torch-cu12",
    ],
    num_structures="random_100",
)

if "error" in results.get("metrics", {}):
    print(f"Error: {results['metrics']['error']}")
else:
    print("Benchmark Results:", results)
