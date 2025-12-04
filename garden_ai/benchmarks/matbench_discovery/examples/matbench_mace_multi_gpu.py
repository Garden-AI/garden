"""Test Matbench Discovery benchmark on Anvil HPC.

This script demonstrates running the IS2RE benchmark with a subset of structures
using multi-GPU parallelization on a Globus Compute endpoint.
"""

from garden_ai.benchmarks.matbench_discovery import DatasetSize, MatbenchDiscovery


# Model factory function for MACE
def create_mace_model(device):
    from mace.calculators import mace_mp

    return mace_mp(model="medium", device=device, default_dtype="float64")


results = MatbenchDiscovery.IS2RE.remote(
    endpoint="5aafb4c1-27b2-40d8-a038-a0277611868f",
    walltime="01:00:00",
    scheduler_options={"gpus-per-node": 2, "cores-per-node": 16},
    account="youraccount",
    partition="gpu-debug",
    qos="gpu",
    model_factory=create_mace_model,
    model_packages="mace-torch",
    num_structures=DatasetSize.RANDOM_100,
)

print(results["metrics"])
