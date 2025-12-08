"""Test Matbench Discovery benchmark on Anvil HPC.

This script demonstrates running the IS2RE benchmark with a subset of structures
using multi-GPU parallelization on a Globus Compute endpoint.
"""

from rich import print

from garden_ai.benchmarks.matbench_discovery import MatbenchDiscovery

ANVIL = "5aafb4c1-27b2-40d8-a038-a0277611868f"


def create_mace_model(device):
    from mace.calculators import mace_mp

    return mace_mp(model="medium-mpa-0", device=device, default_dtype="float64")


results = MatbenchDiscovery.IS2RE.remote(
    endpoint=ANVIL,
    user_endpoint_config={
        "scheduler_options": "#SBATCH --gpus-per-node=4\n",
        "walltime": "05:00:00",
        "qos": "gpu",
        "partition": "gpu",
        "account": "cis250461-gpu",
        "cores_per_node": 16,
        "requirements": "",  # 'requirements' is required for Anvil endpoint
    },
    model_factory=create_mace_model,
    model_packages=[
        "mace-torch",
        "cuequivariance",
        "cuequivariance-torch",
        "cuequivariance-ops-torch-cu12",
    ],
    checkpoint_path="~/.garden/benchmarks/matbench_mace-torch_cuequivariance_full_20251208_115719_ed2e47af.json",
)

print(results)
