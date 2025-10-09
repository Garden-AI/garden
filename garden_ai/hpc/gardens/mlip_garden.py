from pathlib import Path
from typing import Any

from globus_compute_sdk import Client

from garden_ai.gardens import Garden
from garden_ai.hpc.executor import EDITH_EP_ID
from garden_ai.hpc.executor import HpcExecutor as EdithExecutor
from garden_ai.hpc.utils import (
    JobStatus,
    check_file_size_and_read,
    decode_if_base64,
    wait_for_task_id,
)
from garden_ai.schemas.garden import GardenMetadata


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
        relaxation_options: dict = {},
        output_path: str | Path | None = None,
    ):
        cloud_mlip_garden = self.client.get_garden("10.26311/cexg-2349")
        models_to_class_name = {
            "mace": "MACE",
            "mattersim": "MATTERSIM",
            "sevennet": "SEVENNET",
        }
        class_name = models_to_class_name.get(model, None)
        cls = getattr(cloud_mlip_garden, class_name, None) if class_name else None
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

        raw_output = cls.relax(raw_input, relax_params=relaxation_options)

        # If an output path is provided, write results to that file
        if output_path is not None:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                f.write(raw_output if isinstance(raw_output, str) else str(raw_output))

        return raw_output

    def validate_relaxation_params(
        self, relax_params: dict | None, frechet_supported: bool = True
    ) -> tuple[dict, str | None]:
        """
        Validate and normalize relaxation parameters.

        Args:
            relax_params: Dictionary of relaxation parameters from user
            frechet_supported: Whether frechet_cell_fire optimizer is supported

        Returns:
            Tuple of (validated_params, error_message). If error_message is not None,
            validated_params should be ignored.
        """
        if relax_params is None:
            relax_params = {}

        # Create a copy to avoid modifying the original
        validated = {}

        # Validate optimizer_type
        optimizer_type = relax_params.get("optimizer_type", "frechet_cell_fire")
        if optimizer_type not in ["frechet_cell_fire", "fire"]:
            return (
                {},
                f"Invalid optimizer_type '{optimizer_type}'. Must be 'frechet_cell_fire' or 'fire'.",
            )

        if optimizer_type == "frechet_cell_fire" and not frechet_supported:
            return (
                {},
                "The 'frechet_cell_fire' optimizer is not supported in this context. Use 'fire' instead.",
            )

        validated["optimizer_type"] = optimizer_type

        # Validate md_flavor
        md_flavor = relax_params.get("md_flavor", "ase_fire")
        if md_flavor not in ["ase_fire", "vv_fire"]:
            return (
                {},
                f"Invalid md_flavor '{md_flavor}'. Must be 'ase_fire' or 'vv_fire'.",
            )
        validated["md_flavor"] = md_flavor

        # Validate frechet-specific parameters
        frechet_only_bool_params = ["hydrostatic_strain", "constant_volume"]
        frechet_only_float_params = ["scalar_pressure"]

        # Handle boolean frechet params
        for param_name in frechet_only_bool_params:
            if param_name in relax_params:
                param_value = relax_params[param_name]

                # If user is using fire but passed frechet-specific params, error
                if optimizer_type == "fire":
                    return (
                        {},
                        f"Parameter '{param_name}' is only valid for 'frechet_cell_fire' optimizer, but 'fire' optimizer was specified.",
                    )

                # Validate boolean type
                if not isinstance(param_value, bool):
                    return (
                        {},
                        f"Parameter '{param_name}' must be a boolean (True/False), got {type(param_value).__name__}.",
                    )
                validated[param_name] = param_value
            elif optimizer_type == "frechet_cell_fire":
                # Set defaults for frechet_cell_fire
                validated[param_name] = False

        # Handle float frechet params
        for param_name in frechet_only_float_params:
            if param_name in relax_params:
                param_value = relax_params[param_name]

                # If user is using fire but passed frechet-specific params, error
                if optimizer_type == "fire":
                    return (
                        {},
                        f"Parameter '{param_name}' is only valid for 'frechet_cell_fire' optimizer, but 'fire' optimizer was specified.",
                    )

                # Validate float type (allow int too, will be converted)
                if not isinstance(param_value, (float, int)):
                    return (
                        {},
                        f"Parameter '{param_name}' must be a number, got {type(param_value).__name__}.",
                    )
                validated[param_name] = float(param_value)
            elif optimizer_type == "frechet_cell_fire":
                # Set defaults for frechet_cell_fire
                if param_name == "scalar_pressure":
                    validated[param_name] = 0.0

        # Copy through other known parameters with their defaults
        other_params = {
            "fmax": 0.05,
            "max_steps": 500,
        }

        for param_name, default_value in other_params.items():
            if param_name in relax_params:
                validated[param_name] = relax_params[param_name]
            else:
                validated[param_name] = default_value

        return validated, None

    def batch_relax(
        self,
        xyz_file_path: str | Path,
        model: str = "mace",
        max_batch_size: int = 200,
        cluster_id: str | None = None,
        job_options: dict[str, str] | None = None,
        relaxation_options: dict[str, str] | None = None,
        task_id_timeout: int = 60,
    ) -> str:
        """
        Simple batch relaxation that processes structures in small batches.

        Args:
            xyz_file_path: Path to XYZ file containing structures
            model: Model to use for relaxation (any torch-sim supported model)
            max_batch_size: Maximum structures per batch
            cluster_id: HPC endpoint ID to use for computation
            job_options: Job configuration options passed to the globus-compute call
            relaxation_options: Additional options passed through to the relaxation function
            task_id_timeout: How long to wait for a task_id from globus-compute before assuming it failed

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

        validated_params, validation_error = self.validate_relaxation_params(
            relaxation_options
        )
        if validation_error:
            raise ValueError(f"Invalid relaxation parameters: {validation_error}")

        # Determine conda environment based on model
        conda_env = (
            "torch-sim-edith-mace"
            if str(model).startswith("mace")
            else "torch-sim-edith"
        )

        env_path = "/home/hholb/.conda/envs/" + conda_env

        # Pass file content and filename directly to the batch relaxation function
        with EdithExecutor(endpoint_id=cluster_id) as ex:
            future = ex.submit(
                _run_batch_computation,
                file_content,
                xyz_path.name,
                model,
                max_batch_size,
                "relaxation",
                validated_params,
                conda_env_path=env_path,
            )
        task_id = wait_for_task_id(future, task_id_timeout)
        return task_id

    def batch_md(
        self,
        xyz_file_path: str | Path,
        model: str = "mace",
        cluster_id: str = EDITH_EP_ID,
        max_batch_size: int = 10,
        options: dict[str, str] | None = None,
        task_id_timeout: int = 60,
    ) -> str:
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

        task_id = wait_for_task_id(future, task_id_timeout)
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
                    stdout=decode_if_base64(job_result.get("stdout", "")),
                    stderr=decode_if_base64(job_result.get("stderr", "")),
                )

            # Successful completion - results are stored in globus-compute
            return JobStatus(
                status="completed",
                results_available=True,
                stdout=decode_if_base64(
                    job_result.get("stdout", "") if isinstance(job_result, dict) else ""
                ),
                stderr=decode_if_base64(
                    job_result.get("stderr", "") if isinstance(job_result, dict) else ""
                ),
            )

        except Exception as e:
            return JobStatus(
                status="unknown",
                error=f"Failed to get job status: {str(e)}",
            )

    def get_results(
        self,
        job_id: str,
        output_path: str | Path | None = None,
        cluster_id: str | None = None,
    ) -> Path:  # type: ignore[override]
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
    relaxation_options,
    dtype_str: str = "float64",
):
    """
    Generic function to run batch computations (relaxation or MD) on multiple
    structures, saving results incrementally with intelligent batching.

    Args:
        xyz_content: String content of XYZ file.
        xyz_filename: Original filename for output naming.
        model: Model for the computation.
        max_batch_size: Maximum number of structures per batch (used as fallback).
        computation_type: 'relaxation' or 'md'.
        relaxation_options: Validated relaxation parameters dict.
        dtype_str: Data type string ('float64' or 'float32').

    Returns:
        The content of the results XYZ file.
    """
    import os
    import uuid
    from io import StringIO
    from pathlib import Path

    # Fix NUMEXPR threading issue early
    os.environ["NUMEXPR_MAX_THREADS"] = "256"

    import torch
    import torch_sim as ts
    from ase.io import read, write  # type: ignore[import-not-found]
    from torch_sim.runners import generate_force_convergence_fn

    # Parse xyz content using StringIO to match _perform_batch_relaxation pattern
    string_file = StringIO(xyz_content)
    all_atoms = read(string_file, index=":", format="extxyz")
    num_structures = len(all_atoms)

    # Create results file path
    input_path = Path(xyz_filename)
    suffix = "relaxed" if computation_type == "relaxation" else "md"
    results_filename = f"{input_path.stem}_{suffix}_{uuid.uuid4().hex[:8]}.xyz"
    results_path = input_path.parent / results_filename

    print(f"📝 Results will be saved to: {results_path}")
    print(f"🚀 Starting batch {computation_type} of {num_structures} structures...")

    # Set up device and resolve dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype_str == "float64":
        torch_dtype = torch.float64
    elif dtype_str == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype string: {dtype_str}")

    # Extract relaxation parameters
    if computation_type == "relaxation":
        optimizer_type = relaxation_options.get("optimizer_type", "frechet_cell_fire")
        md_flavor = relaxation_options.get("md_flavor", "ase_fire")
        fmax = relaxation_options.get("fmax", 0.05)
        max_steps = relaxation_options.get("max_steps", 500)

        if optimizer_type == "frechet_cell_fire":
            optimizer_builder = ts.frechet_cell_fire
            optimize_kwargs = {
                "convergence_fn": generate_force_convergence_fn(force_tol=fmax)
            }
            # Add frechet-specific options
            frechet_kwargs = {}
            if "hydrostatic_strain" in relaxation_options:
                frechet_kwargs["hydrostatic_strain"] = relaxation_options[
                    "hydrostatic_strain"
                ]
            if "constant_volume" in relaxation_options:
                frechet_kwargs["constant_volume"] = relaxation_options[
                    "constant_volume"
                ]
            if "scalar_pressure" in relaxation_options:
                frechet_kwargs["scalar_pressure"] = relaxation_options[
                    "scalar_pressure"
                ]
        elif optimizer_type == "fire":
            optimizer_builder = ts.optimizers.fire
            optimize_kwargs = {}
            frechet_kwargs = {}
        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")

    def load_torch_sim_model(model_name: str, device: str = "cpu", dtype=None):
        """Load a torch-sim model by name."""
        if dtype is None:
            dtype = torch.float32
        print(f"🔧 Loading {model_name} model on {device}...")
        if model_name.startswith("mace"):
            from mace.calculators import mace_mp
            from torch_sim.models.mace import MaceModel

            # Get the raw model directly with weights_only=False for checkpoint loading
            raw_model = mace_mp(
                model="medium-mpa-0",
                device=device,
                enable_cueq=True,
                default_dtype="float64",
                return_raw_model=True,
            )
            return MaceModel(model=raw_model, device=device)
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
        elif model_name == "sevennet":
            from sevenn.calculator import SevenNetCalculator
            from torch_sim.models.sevennet import SevenNetModel

            modal = "mpa"
            sevennet_calculator = SevenNetCalculator(
                model="7net-mf-ompa", modal=modal, device=device
            )

            return SevenNetModel(
                model=sevennet_calculator.model, modal=modal, device=device
            )
        elif model_name == "mattersim":
            from mattersim.forcefield.potential import Potential
            from torch_sim.models.mattersim import MatterSimModel

            potential = Potential.from_checkpoint(
                load_path="mattersim-v1.0.0-5m", device=str(device)
            )
            # Pass the loaded potential into the official torch-sim wrapper
            return MatterSimModel(model=potential)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    torch_sim_model = load_torch_sim_model(model, device, torch_dtype)

    # Extract material IDs (create if not present) and atom counts
    mat_ids = []
    for idx, atoms in enumerate(all_atoms):
        if "material_id" in atoms.info:
            mat_ids.append(atoms.info["material_id"])
        else:
            mat_id = f"material_{idx}"
            atoms.info["material_id"] = mat_id
            mat_ids.append(mat_id)

    atom_counts = [len(atoms) for atoms in all_atoms]

    print(f"  - Found {len(all_atoms)} materials")
    print(f"  - Atom count range: {min(atom_counts)} - {max(atom_counts)}")

    # Create lists for each size category with (index, atoms, mat_id) tuples
    small_materials = []  # <10 atoms
    medium_materials = []  # 10-19 atoms
    large_materials = []  # 20+ atoms

    for idx, (atoms, mat_id, n_atoms) in enumerate(
        zip(all_atoms, mat_ids, atom_counts)
    ):
        material_data = (idx, atoms, mat_id)
        if n_atoms < 10:
            small_materials.append(material_data)
        elif n_atoms < 20:
            medium_materials.append(material_data)
        else:
            large_materials.append(material_data)

    print("\n📊 Size distribution:")
    print(f"  - Small (<10 atoms): {len(small_materials)} materials")
    print(f"  - Medium (10-19 atoms): {len(medium_materials)} materials")
    print(f"  - Large (20+ atoms): {len(large_materials)} materials")

    # Define batch sizes for each category
    batch_sizes = {"small": 200, "medium": 100, "large": 25}

    # Track results for proper ordering
    all_results = []
    has_written = False

    def process_category(materials, category_name, batch_size):
        """Process all materials in a size category."""
        nonlocal has_written

        if not materials:
            print("nothing in " + category_name)
            return

        n_batches = (len(materials) + batch_size - 1) // batch_size
        print(
            f"\n🔄 Processing {category_name} materials: {len(materials)} materials in {n_batches} batches"
        )

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(materials))
            batch_materials = materials[start_idx:end_idx]

            batch_atoms = [item[1] for item in batch_materials]
            batch_mat_ids = [item[2] for item in batch_materials]
            batch_original_indices = [item[0] for item in batch_materials]

            print(
                f"  - Batch {batch_idx + 1}/{n_batches}: {len(batch_atoms)} materials"
            )

            # Initialize state
            initial_state = ts.initialize_state(
                batch_atoms, device=device, dtype=torch_dtype
            )

            # Run computation
            if computation_type == "relaxation":

                def optimizer_callable(model, **_kwargs):
                    if optimizer_type == "frechet_cell_fire":
                        return optimizer_builder(
                            model, md_flavor=md_flavor, **frechet_kwargs
                        )
                    else:
                        return optimizer_builder(model, md_flavor=md_flavor)

                result_state = ts.optimize(
                    system=initial_state,
                    model=torch_sim_model,
                    optimizer=optimizer_callable,
                    max_steps=max_steps,
                    **optimize_kwargs,
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
            energies_tensor = result_state.energy

            # Ensure energies are always in a list
            if energies_tensor.dim() == 0:
                # This is a 0-d tensor, so convert to a list with one item
                final_energies = [energies_tensor.item()]
            else:
                # This is a 1-d or higher tensor, tolist() works correctly
                final_energies = energies_tensor.cpu().tolist()

            # Also ensure the atoms list is always a list for consistency
            if not isinstance(result_atoms_list, list):
                result_atoms_list = [result_atoms_list]

            batch_results = [
                {
                    "material_id": mat_id,
                    "final_energy": energy,
                    "relaxed_atoms": atoms,
                    "original_index": orig_idx,
                }
                for mat_id, atoms, energy, orig_idx in zip(
                    batch_mat_ids,
                    result_atoms_list,
                    final_energies,
                    batch_original_indices,
                )
            ]
            all_results.extend(batch_results)

            # Write results incrementally in case job gets killed before it finishes
            print(f"💾 Saving {len(result_atoms_list)} structures to {results_path}...")
            for atoms, energy, mat_id in zip(
                result_atoms_list, final_energies, batch_mat_ids
            ):
                atoms.info["energy"] = energy
                atoms.info["final_energy"] = energy
                atoms.info["material_id"] = mat_id
                atoms.info["model"] = model
                if computation_type == "relaxation":
                    atoms.info["relaxed"] = True
                write(results_path, atoms, append=has_written)
                if not has_written:
                    has_written = True

    # Process each category
    process_category(small_materials, "small", batch_sizes["small"])
    process_category(medium_materials, "medium", batch_sizes["medium"])
    process_category(large_materials, "large", batch_sizes["large"])

    # Read final results content
    with open(results_path, "r") as f:
        results_content = f.read()

    print(
        f"\n✅ {computation_type.capitalize()} complete! Returning results for {num_structures} structures"
    )
    return results_content
