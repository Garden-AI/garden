from dataclasses import dataclass
from pathlib import Path
from typing import Any

from globus_compute_sdk import Client

from garden_ai.gardens import Garden
from garden_ai.hpc_executors.edith_executor import EDITH_EP_ID, EdithExecutor
from garden_ai.hpc_gardens.utils import check_file_size_and_read
from garden_ai.schemas.garden import GardenMetadata


@dataclass
class JobStatus:
    """Status information for a batch job."""

    status: str  # "pending" | "running" | "completed" | "failed" | "unknown"
    results_available: bool = False
    error: str | None = None
    stdout: str = ""
    stderr: str = ""


class MLIPGarden(Garden):
    """A Garden that uses EdithExecutor for MLIP computations."""

    def __init__(self, client, doi: str):
        self.client = client
        self.jobs: dict[str, dict[str, Any]] = {}

        metadata = GardenMetadata(
            doi=doi,
            authors=["MLIP Team"],
            title="MLIP Garden",
            description="Machine Learning Interatomic Potentials for materials science, runs on HPCs via globus-compute.",
            doi_is_draft=False,
        )
        super().__init__(metadata=metadata)

    def relax(
        self,
        xyz_file_path: str | Path | None = None,
        xyz_string: str | None = None,
        model: str = "mace",
        options: dict = {},
    ):
        cloud_mlip_garden = self.client.get_garden("10.26311/qs92-3t67")
        models_to_class_name = {
            "mace": "MACE",
            "mattersim": "MATTERSIM",
            "sevennet": "SEVENNET",
        }
        class_name = models_to_class_name.get(model, None)
        cls = getattr(cloud_mlip_garden, class_name, None)
        if cls is None:
            raise ValueError(f"Model {model} not found")
        # If we're given a file path, read the file into a string
        if xyz_string is not None:
            raw_input = xyz_string
        elif xyz_file_path is not None:
            try:
                with open(xyz_file_path, "r") as f:
                    raw_input = f.read()
            except Exception as e:
                raise ValueError(f"Could not read {xyz_file_path}: {e}")
        else:
            raise ValueError("No input provided")

        raw_output = cls.relax(raw_input)
        return raw_output

    def batch_relax(
        self,
        xyz_file_path: str | Path,
        model: str = "mace-mp-0",
        max_batch_size: int = 10,
        cluster_id: str | None = None,
        options: dict[str, str] | None = None,
    ) -> str:
        """
        Simple batch relaxation that processes structures in small batches.

        Args:
            xyz_file_path: Path to XYZ file containing structures
            model: Model to use for relaxation (any torch-sim supported model)
            max_batch_size: Maximum structures per batch
            cluster_id: HPC endpoint ID to use for computation
            options: Additional options

        Returns:
            Future returned by globus-compute
        """
        from pathlib import Path

        # Check file size and read content locally
        xyz_path = Path(xyz_file_path)
        try:
            file_content = check_file_size_and_read(xyz_path)
        except (FileNotFoundError, ValueError) as e:
            raise RuntimeError(f"Failed to read XYZ file: {e}")

        ex = EdithExecutor(endpoint_id=cluster_id)

        # Pass file content and filename directly to the batch relaxation function
        future = ex.submit(
            _run_batch_computation,
            file_content,
            xyz_path.name,
            model,
            max_batch_size,
            "relaxation",
        )
        task_id = future.task_id

        # TODO: this could be an infinite loop if we never get the task id
        while task_id is None:
            task_id = future.task_id

        return task_id

    def batch_md(
        self,
        xyz_file_path: str | Path,
        model: str = "mace-mp-0",
        cluster_id: str = EDITH_EP_ID,
        max_batch_size: int = 10,
        options: dict[str, str] | None = None,
    ):
        xyz_path = Path(xyz_file_path)
        try:
            file_content = check_file_size_and_read(xyz_file_path)
        except (FileNotFoundError, ValueError) as e:
            raise RuntimeError(f"Failed to read XYZ file: {e}")

        ex = EdithExecutor(endpoint_id=cluster_id)
        future = ex.submit(
            _run_batch_computation,
            file_content,
            xyz_path.name,
            model,
            max_batch_size,
            "md",
        )

        task_id = future.task_id
        while task_id is None:
            task_id = future.task_id

        return task_id

    def get_job_status(self, job_id: str) -> JobStatus:
        """
        Get comprehensive status information for a submitted batch job.

        Args:
            job_id: Job ID returned by batch_relax

        Returns:
            JobStatus dataclass with status information
        """
        client = Client()
        task_info = client.get_task(job_id)

        # Extract basic status
        if task_info.get("pending", True):
            return JobStatus(status="pending")

        # Get the actual result to determine if completed or failed
        try:
            job_result = client.get_result(job_id)

            if isinstance(job_result, dict) and "error" in job_result:
                return JobStatus(
                    status="failed",
                    error=job_result["error"],
                    stdout=job_result.get("stdout", ""),
                    stderr=job_result.get("stderr", ""),
                )

            # Successful completion - results are stored in globus-compute
            return JobStatus(
                status="completed",
                results_available=True,
                stdout=(
                    job_result.get("stdout", "") if isinstance(job_result, dict) else ""
                ),
                stderr=(
                    job_result.get("stderr", "") if isinstance(job_result, dict) else ""
                ),
            )

        except Exception as e:
            return JobStatus(
                status="unknown",
                error=f"Failed to get job status: {str(e)}",
            )

    def get_results(
        self, job_id: str, output_path: str | Path = None, cluster_id: str | None = None  # type: ignore[assignment]
    ) -> Path:
        """
        Download the results of a completed batch job to local machine.

        Args:
            job_id: Job ID returned by batch_relax
            output_path: Local path to save results (default: ./results_{job_id}.xyz)
            cluster_id: HPC endpoint ID (if different from original job)

        Returns:
            Path to local results file

        Raises:
            RuntimeError: If job is not completed or failed
        """
        from pathlib import Path

        # Check job status first
        status_info = self.get_job_status(job_id)

        if status_info.status == "pending":
            raise RuntimeError(f"Job {job_id} is still pending")
        elif status_info.status == "running":
            raise RuntimeError(f"Job {job_id} is still running")
        elif status_info.status == "failed":
            raise RuntimeError(f"Job {job_id} failed: {status_info.error}")
        elif status_info.status != "completed":
            raise RuntimeError(f"Job {job_id} has unknown status: {status_info.status}")

        if not status_info.results_available:
            raise RuntimeError(f"No results available for completed job {job_id}")

        # Set up output path
        if output_path is None:
            output_path = Path(f"./results_{job_id}.xyz")
        else:
            output_path = Path(output_path)

        # Set up output path and ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"📁 Saving to local file: {output_path}")

        client = Client()
        job_result = client.get_result(job_id)

        # Handle result decoding if needed (results are stored as encoded data)
        if isinstance(job_result, dict) and "raw_data" in job_result:
            ex = EdithExecutor()  # Need this for decoding
            file_content = ex.decode_result_data(job_result["raw_data"])
        else:
            file_content = job_result

        # Ensure we have string content
        if not isinstance(file_content, str):
            raise RuntimeError(f"Expected string content, got {type(file_content)}")

        # Write entire file to local path
        with open(output_path, "w") as f:
            f.write(file_content)

        print(f"✅ Successfully saved results to {output_path}")
        return output_path


def _run_batch_computation(
    xyz_content,
    xyz_filename,
    model,
    max_batch_size,
    computation_type,
):
    """
    Generic function to run batch computations (relaxation or MD) on multiple
    structures, saving results incrementally.

    Args:
        xyz_content: String content of XYZ file.
        xyz_filename: Original filename for output naming.
        model: Model for the computation.
        max_batch_size: Number of structures per batch.
        computation_type: 'relaxation' or 'md'.

    Returns:
        The content of the results XYZ file.
    """
    import tempfile
    import uuid
    from pathlib import Path

    import ase  # noqa:
    import numpy as np  # noqa:
    import torch
    import torch_sim as ts
    from ase.io import read, write  # type: ignore[import-not-found]

    # Write XYZ content to a temporary file on remote endpoint
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xyz", delete=False
    ) as temp_file:
        temp_file.write(xyz_content)
        temp_xyz_path = temp_file.name

    # Read structures from temporary xyz file
    all_atoms = read(temp_xyz_path, index=":")
    num_structures = len(all_atoms)

    # Create results file path
    input_path = Path(xyz_filename)
    suffix = "relaxed" if computation_type == "relaxation" else "md"
    results_filename = f"{input_path.stem}_{suffix}_{uuid.uuid4().hex[:8]}.xyz"
    results_path = input_path.parent / results_filename

    print(f"📝 Results will be saved to: {results_path}")
    print(
        f"🚀 Starting batch {computation_type} of {num_structures} structures in batches of {max_batch_size}..."
    )

    # Set up device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    def load_torch_sim_model(model_name: str, device: str = "cpu", dtype=None):
        """Load a torch-sim model by name."""
        if dtype is None:
            dtype = torch.float32
        print(f"🔧 Loading {model_name} model on {device}...")
        if model_name.startswith("mace"):
            from mace.calculators.foundations_models import mace_mp, mace_off
            from torch_sim.models.mace import MaceModel

            if "off" in model_name:
                size = model_name.split("-")[-1] if "-" in model_name else "medium"
                loaded_model = mace_off(
                    model=size,
                    return_raw_model=True,
                    default_dtype=dtype,
                    device=device,
                )
            else:
                variant = model_name.replace("mace-", "").replace("mp-", "")
                if variant == "0":
                    variant = "medium"
                loaded_model = mace_mp(
                    model=variant if variant else "medium",
                    return_raw_model=True,
                    default_dtype=dtype,
                    device=device,
                )
            return MaceModel(
                model=loaded_model,
                device=device,
                compute_forces=True,
                compute_stress=True,
                dtype=dtype,
            )
        elif model_name in ["soft-sphere", "lennard-jones", "morse"]:
            if model_name == "soft-sphere":
                from torch_sim.models.soft_sphere import SoftSphereModel

                return SoftSphereModel(device=device, dtype=dtype)
            elif model_name == "lennard-jones":
                from torch_sim.models.lennard_jones import LennardJonesModel

                return LennardJonesModel(
                    sigma=2.0, epsilon=0.1, device=device, dtype=dtype
                )
            elif model_name == "morse":
                from torch_sim.models.morse import MorseModel

                return MorseModel(device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    torch_sim_model = load_torch_sim_model(model, device, dtype)

    has_written = False
    for i in range(0, num_structures, max_batch_size):
        batch_atoms = all_atoms[i : i + max_batch_size]
        print(
            f"⚙️ Processing batch {i//max_batch_size + 1}/{(num_structures + max_batch_size - 1)//max_batch_size}..."
        )

        initial_state = ts.initialize_state(batch_atoms, device=device, dtype=dtype)

        if computation_type == "relaxation":
            result_state = ts.optimize(
                system=initial_state,
                model=torch_sim_model,
                optimizer=ts.frechet_cell_fire,
                max_steps=200,
                autobatcher=True,
                convergence_fn=ts.runners.generate_force_convergence_fn(
                    force_tol=0.05, include_cell_forces=False
                ),
            )
        elif computation_type == "md":
            result_state = ts.integrate(
                system=initial_state,
                model=torch_sim_model,
                integrator=ts.integrators.nvt_langevin,
                n_steps=50,
                timestep=0.002,
                temperature=1000,
                autobatcher=True,
            )
        else:
            raise ValueError(f"Unknown computation_type: {computation_type}")

        result_atoms_list = result_state.to_atoms()
        final_energies = result_state.energy.cpu().tolist()

        print(f"💾 Saving {len(result_atoms_list)} structures to {results_path}...")
        for atoms, energy in zip(result_atoms_list, final_energies):
            atoms.info["energy"] = energy
            atoms.info["model"] = model
            if computation_type == "relaxation":
                atoms.info["relaxed"] = True
            write(results_path, atoms, append=has_written)
            if not has_written:
                has_written = True

    with open(results_path, "r") as f:
        results_content = f.read()

    return results_content
