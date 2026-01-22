#!/usr/bin/env python3
"""Matbench Discovery Benchmark - EquiformerV2 Example"""

from garden_ai.benchmarks.matbench_discovery import MatbenchDiscovery


def create_equiformerv2_model(device):
    from fairchem.core.calculate.ase_calculator import Calculator  # type: ignore

    # Use pre-trained checkpoint - will auto-download from HuggingFace
    return Calculator(
        model_name="EquiformerV2-31M-S2EF-OC20-All+MD", cpu=(device == "cpu")
    )


# Run S2EFS task (structure to energy/forces/stress)
output = MatbenchDiscovery.S2EFS.remote(
    endpoint="anvil",
    account="your-account-here",
    model_factory=create_equiformerv2_model,
    model_packages="fairchem-core",
    num_structures="random_10k",
)

if "error" in output.get("metrics", {}):
    print(f"Error: {output['metrics']['error']}")
else:
    print("Benchmark Results:", output.get("metrics"))
