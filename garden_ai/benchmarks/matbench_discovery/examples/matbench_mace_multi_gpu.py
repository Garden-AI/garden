"""Test Matbench Discovery benchmark on Anvil HPC.

This script demonstrates running the IS2RE benchmark with a subset of structures
using multi-GPU parallelization on a Globus Compute endpoint.
"""

from garden_ai.benchmarks.matbench_discovery import MatbenchDiscovery

ANVIL = "5aafb4c1-27b2-40d8-a038-a0277611868f"


# Model factory function for MACE
def create_mace_model(device):
    from mace.calculators import mace_mp

    return mace_mp(model="medium", device=device, default_dtype="float64")


results = MatbenchDiscovery.IS2RE.remote(
    endpoint=ANVIL,
    user_endpoint_config={
        "scheduler_options": "#SBATCH --gpus-per-node=2\n",
        "walltime": 3600,
        "qos": "gpu",
        "partition": "gpu-debug",
        "account": "cis250461-gpu",
        "cores_per_node": 16,
        "mem_per_node": 32,
        "requirements": "",  # 'requirements' is required for Anvil endpoint
    },
    model_factory=create_mace_model,
    model_packages=[
        "mace-torch",
        "cuequivariance",
        "cuequivariance-torch",
        "cuequivariance-ops-torch-cu12",
    ],
    num_structures="random_100",
)

print(results["metrics"])
