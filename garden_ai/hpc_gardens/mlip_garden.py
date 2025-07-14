from pathlib import Path
from typing import Any

try:
    import ase
    import numpy as np
    from ase.io import read

    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    ase = None
    np = None
    read = None

from garden_ai.gardens import Garden
from garden_ai.hpc_executors.edith_executor import EdithExecutor
from garden_ai.schemas.garden import GardenMetadata


class MLIPGarden(Garden):
    """A Garden that uses EdithExecutor for MLIP computations."""

    def __init__(self, client, doi: str):
        if not ASE_AVAILABLE:
            raise ImportError(
                "To use MLIP Garden functionality, install garden-ai with the 'mlip' extra: "
                "pip install garden-ai[mlip]"
            )
        self.client = client
        self.jobs: dict[str, dict[str, Any]] = {}

        metadata = GardenMetadata(
            doi=doi,
            authors=["MLIP Team"],
            title="MLIP Garden",
            description="Machine Learning Interatomic Potentials for materials science",
            doi_is_draft=False,
        )
        super().__init__(metadata=metadata)

    def run_relaxation(
        self, atoms, model: str = "mace-mp-0", cluster_id: str | None = None
    ):
        """
        Run relaxation on-demand. This is intended for small inputs and testing.
        For large inputs use the batch_relax flow.

        Args:
            atoms: ASE Atoms object to relax
            model: Model to use for relaxation (any torch-sim supported model)
            cluster_id: HPC endpoint ID to use for computation

        Returns:
            Future object from the executor
        """
        ex = EdithExecutor(endpoint_id=cluster_id)

        # Convert atoms to serializable dict
        atoms_dict = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in atoms.todict().items()
        }

        # Submit the function
        fut = ex.submit(_run_single_relaxation, atoms_dict, model)
        raw_result = fut.result()
        decoded = ex.decode_result_data(raw_result.get("raw_data"))
        return ase.Atoms.fromdict(decoded)

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
            Job ID for tracking
        """
        import uuid
        from pathlib import Path

        # Read structures locally
        xyz_path = Path(xyz_file_path)
        if not xyz_path.exists():
            raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

        print(f"📖 Reading structures from {xyz_path}")
        atoms_list = read(xyz_path, index=":")
        print(f"  - Found {len(atoms_list)} structures")

        # Generate job ID
        job_id = f"batch_relax_{uuid.uuid4().hex[:8]}"

        # Convert atoms to serializable format
        atoms_dicts = []
        for i, atoms in enumerate(atoms_list):
            atoms_dict = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in atoms.todict().items()
            }
            atoms_dict["_index"] = i  # Keep track of original order
            atoms_dicts.append(atoms_dict)

        # Submit batch job
        executor = EdithExecutor(endpoint_id=cluster_id)
        print(f"🚀 Submitting batch job {job_id} to HPC...")
        future = executor.submit(
            _run_batch_relaxation, atoms_dicts, model, max_batch_size
        )

        # Store job info
        self.jobs[job_id] = {
            "future": future,
            "model": model,
            "num_structures": len(atoms_list),
            "input_file": str(xyz_path),
            "options": options or {},
            "status": "submitted",
        }

        print(f"✅ Batch job {job_id} submitted successfully!")
        return job_id

    def get_job_status(self, job_id: str) -> str:
        """
        Get the status of a submitted batch job.

        Args:
            job_id: Job ID returned by batch_relax

        Returns:
            Job status string: "submitted", "running", "completed", "failed", or "unknown"
        """
        if job_id not in self.jobs:
            return "unknown"

        job_info = self.jobs[job_id]
        future = job_info["future"]

        try:
            # Check if the future is done
            if future.done():
                # Try to get the result to check for errors
                try:
                    result = future.result()
                    if "error" in result:
                        job_info["status"] = "failed"
                        job_info["error"] = result["error"]
                    else:
                        job_info["status"] = "completed"
                        job_info["result"] = result
                except Exception as e:
                    job_info["status"] = "failed"
                    job_info["error"] = str(e)
            else:
                # Job is still running
                job_info["status"] = "running"

        except Exception as e:
            job_info["status"] = "failed"
            job_info["error"] = f"Status check failed: {str(e)}"

        return job_info["status"]

    def get_results(self, job_id: str):
        """
        Get the results of a completed batch job.

        Args:
            job_id: Job ID returned by batch_relax

        Returns:
            Dictionary containing relaxed structures and energies, or None if job not complete
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        job_info = self.jobs[job_id]

        # Check if job is complete
        status = self.get_job_status(job_id)
        if status != "completed":
            if status == "failed":
                error_msg = job_info.get("error", "Unknown error")
                raise RuntimeError(f"Job {job_id} failed: {error_msg}")
            else:
                return None  # Job not complete yet

        # Get the result from the future
        try:
            result = job_info["future"].result()

            # Parse the result if it contains ASE data
            if "raw_data" in result:
                executor = EdithExecutor()
                parsed_result = executor.decode_result_data(result["raw_data"])
                atoms = [ase.Atoms.fromdict(d) for d in parsed_result]
                return atoms
            else:
                # Return raw result if no ASE data
                return result

        except Exception as e:
            raise RuntimeError(f"Failed to retrieve results for job {job_id}: {str(e)}")

    def cleanup_job(self, job_id: str):
        """
        Clean up resources associated with a completed job.

        Args:
            job_id: Job ID to clean up
        """
        if job_id in self.jobs:
            # Could add cleanup of staged files here if needed
            del self.jobs[job_id]
            print(f"🧹 Cleaned up job {job_id}")

    def list_jobs(self) -> dict[str, dict]:
        """
        List all tracked jobs and their current status.

        Returns:
            Dictionary mapping job IDs to job info
        """
        job_statuses = {}
        for job_id in self.jobs:
            job_info = self.jobs[job_id].copy()
            job_info["current_status"] = self.get_job_status(job_id)
            # Don't include the future object in the output
            job_info.pop("future", None)
            job_statuses[job_id] = job_info

        return job_statuses


def run_chunk_size_binned(
    chunk_index: int,
    chunk_dir_name: str = "smallest_chunks",
    mace_model=None,
    device=None,
    base_dir: str | None = None,
):
    """
    Runs the relaxation workflow with simple size-based batching.

    Shamelessly pulled from:
    https://github.com/Garden-AI/uploadathon/blob/main/matbench/matbench_mace.py

    Batching strategy:
    - Materials with <10 atoms: 80 per batch
    - Materials with 10-19 atoms: 40 per batch
    - Materials with 20+ atoms: 16 per batch

    Args:
        chunk_index: Index of the chunk to process
        chunk_dir_name: Name of the directory containing chunks
        mace_model: The MACE model to use for relaxation
        device: Device to run computations on
        base_dir: Base directory for data (defaults to user home)
    """
    import json
    from os.path import expanduser
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    import torch_sim as ts
    from ase.io import read

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    def generate_force_convergence_fn(force_tol=0.05):
        """Generate a convergence function based on force tolerance."""

        def convergence_fn(state):
            forces = state.force
            max_force = torch.max(torch.norm(forces, dim=-1))
            return max_force < force_tol

        return convergence_fn

    # Use user home directory if no base_dir provided
    if base_dir is None:
        base_dir = expanduser("~")

    chunk_path = Path(base_dir) / chunk_dir_name / str(chunk_index)
    results_file = chunk_path / "results.parquet"

    if results_file.exists():
        print(
            f"✅ Chunk {chunk_index} already processed. Results found at {results_file}. Skipping."
        )
        return

    print(f"🎬 Starting size-batched run for chunk {chunk_index}...")

    # Read all atoms and get their info
    initial_atoms_list = read(chunk_path / "atoms.traj", index=":")
    mat_ids = [
        atoms.info.get("mat_id", f"material_{i}")
        for i, atoms in enumerate(initial_atoms_list)
    ]
    atom_counts = [len(atoms) for atoms in initial_atoms_list]

    print(f"  - Found {len(initial_atoms_list)} materials")
    print(f"  - Atom count range: {min(atom_counts)} - {max(atom_counts)}")

    # Create lists for each size category with (index, atoms, mat_id) tuples
    small_materials = []  # <10 atoms
    medium_materials = []  # 10-19 atoms
    large_materials = []  # 20+ atoms

    for idx, (atoms, mat_id, n_atoms) in enumerate(
        zip(initial_atoms_list, mat_ids, atom_counts)
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
    batch_sizes = {"small": 80, "medium": 40, "large": 16}

    # Process each size category
    all_results = []

    # Helper function to process a category
    def process_category(materials, category_name, batch_size):
        """Process all materials in a size category."""
        if not materials:
            return

        n_batches = (len(materials) + batch_size - 1) // batch_size
        print(
            f"\n🔄 Processing {category_name} materials: {len(materials)} materials in {n_batches} batches"
        )

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(materials))
            batch_materials = materials[start_idx:end_idx]

            # Extract atoms and mat_ids for this batch
            batch_atoms = [item[1] for item in batch_materials]
            batch_mat_ids = [item[2] for item in batch_materials]
            batch_original_indices = [item[0] for item in batch_materials]

            print(
                f"  - Batch {batch_idx + 1}/{n_batches}: {len(batch_atoms)} materials"
            )

            # Run relaxation
            initial_state = ts.initialize_state(
                batch_atoms, device=device, dtype=torch.float64
            )
            optimizer_builder = ts.frechet_cell_fire

            def optimizer_callable(model, **kwargs):
                return optimizer_builder(model, md_flavor="ase_fire", **kwargs)

            relaxed_state = ts.optimize(
                system=initial_state,
                model=mace_model,
                optimizer=optimizer_callable,
                max_steps=500,
                convergence_fn=generate_force_convergence_fn(force_tol=0.05),
            )

            # Collect results
            relaxed_atoms_list = relaxed_state.to_atoms()
            final_energies = relaxed_state.energy.cpu().tolist()

            batch_results = [
                {
                    "material_id": mat_id,
                    "final_energy": energy,
                    "relaxed_atoms_json": json.dumps(atoms.todict(), cls=NumpyEncoder),
                    "original_index": orig_idx,  # Keep track of original ordering
                }
                for mat_id, atoms, energy, orig_idx in zip(
                    batch_mat_ids,
                    relaxed_atoms_list,
                    final_energies,
                    batch_original_indices,
                )
            ]
            all_results.extend(batch_results)

    # Process each category
    process_category(small_materials, "small", batch_sizes["small"])
    process_category(medium_materials, "medium", batch_sizes["medium"])
    process_category(large_materials, "large", batch_sizes["large"])

    # Sort results back to original order before saving
    all_results.sort(key=lambda x: x["original_index"])

    # Remove the temporary original_index field
    for result in all_results:
        del result["original_index"]

    print(f"\n✅ Chunk {chunk_index} processing complete")
    print(f"  - Total materials processed: {len(all_results)}")

    # Save results
    df_results = pd.DataFrame(all_results).set_index("material_id")
    print(f"💾 Saving results to {results_file}...")
    # Ensure the directory exists
    results_file.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_parquet(results_file)
    print("  - Save complete!")


def _run_single_relaxation(atoms_dict, model: str = "mace-mp-0"):
    """
    Standalone function to run relaxation on a single structure.

    Args:
        atoms_dict: Dictionary representation of ASE atoms object
        model: Model to use for relaxation

    Returns:
        Dictionary representation of relaxed atoms
    """
    import ase
    import numpy as np
    import torch
    import torch_sim as ts

    # Convert dict back to atoms
    atoms = ase.Atoms.fromdict(atoms_dict)

    # Set up device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # Load the specified model - inline function for remote execution
    def load_torch_sim_model(model_name: str, device: str = "cpu", dtype=None):
        """Load a torch-sim model by name."""
        if dtype is None:
            dtype = torch.float32

        print(f"🔧 Loading {model_name} model on {device}...")

        # MACE models
        if model_name.startswith("mace"):
            from mace.calculators.foundations_models import mace_mp, mace_off
            from torch_sim.models.mace import MaceModel, MaceUrls  # noqa:

            if "off" in model_name:
                # MACE-OFF models for organic molecules
                size = model_name.split("-")[-1] if "-" in model_name else "medium"
                loaded_model = mace_off(
                    model=size,
                    return_raw_model=True,
                    default_dtype=dtype,
                    device=device,
                )
            else:
                # MACE-MP models
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

        # Classical potentials
        elif model_name in ["soft-sphere", "lennard-jones", "morse"]:
            if model_name == "soft-sphere":
                from torch_sim.models.soft_sphere import SoftSphereModel

                return SoftSphereModel(device=device, dtype=dtype)
            elif model_name == "lennard-jones":
                from torch_sim.models.lennard_jones import LennardJonesModel

                return LennardJonesModel(
                    sigma=2.0,  # Å, interaction distance
                    epsilon=0.1,  # eV, interaction strength
                    device=device,
                    dtype=dtype,
                )
            elif model_name == "morse":
                from torch_sim.models.morse import MorseModel

                return MorseModel(device=device, dtype=dtype)

        # FairChem models (example - would need actual implementation)
        elif model_name.startswith("fairchem"):
            raise NotImplementedError(
                f"FairChem models not yet implemented: {model_name}"
            )

        # SevenNet models (example - would need actual implementation)
        elif model_name.startswith("sevennet"):
            raise NotImplementedError(
                f"SevenNet models not yet implemented: {model_name}"
            )

        # ORB models (example - would need actual implementation)
        elif model_name.startswith("orb"):
            raise NotImplementedError(f"ORB models not yet implemented: {model_name}")

        else:
            raise ValueError(
                f"Unknown model: {model_name}. Supported models: mace-*, mace-off-*, soft-sphere, lennard-jones, morse"
            )

    torch_sim_model = load_torch_sim_model(model, device, dtype)

    print(f"🚀 Running relaxation on {len(atoms)} atoms...")

    # Run relaxation
    final_state = ts.optimize(
        system=atoms, model=torch_sim_model, optimizer=ts.unit_cell_fire
    )

    # Convert result back to dict
    result_atoms = final_state.to_atoms()

    # Handle case where result is a single atoms object or list
    if isinstance(result_atoms, list):
        result_dicts = [atoms.todict() for atoms in result_atoms]
    else:
        result_dicts = [result_atoms.todict()]

    # Convert numpy arrays to lists for serialization
    for d in result_dicts:
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()

    print("✅ Relaxation complete!")
    return result_dicts[0] if len(result_dicts) == 1 else result_dicts


def _run_batch_relaxation(atoms_dicts, model="mace-mp-0", max_batch_size=10):
    """
    Run batch relaxation on multiple structures using torch-sim's autobatcher.

    Args:
        atoms_dicts: List of dictionary representations of ASE atoms objects
        model: Model to use for relaxation (any torch-sim supported model)
        max_batch_size: Maximum structures to process at once (legacy parameter, autobatcher handles this)

    Returns:
        List of relaxed atoms dictionaries
    """
    import ase
    import numpy as np
    import torch
    import torch_sim as ts

    print(
        f"🚀 Starting batch relaxation of {len(atoms_dicts)} structures using torch-sim autobatcher..."
    )

    # Set up device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64  # Use float64 for better numerical stability

    # Load the specified model - inline function for remote execution
    def load_torch_sim_model(model_name: str, device: str = "cpu", dtype=None):
        """Load a torch-sim model by name."""
        if dtype is None:
            dtype = torch.float32

        print(f"🔧 Loading {model_name} model on {device}...")

        # MACE models
        if model_name.startswith("mace"):
            from mace.calculators.foundations_models import mace_mp, mace_off
            from torch_sim.models.mace import MaceModel, MaceUrls  # noqa:

            if "off" in model_name:
                # MACE-OFF models for organic molecules
                size = model_name.split("-")[-1] if "-" in model_name else "medium"
                loaded_model = mace_off(
                    model=size,
                    return_raw_model=True,
                    default_dtype=dtype,
                    device=device,
                )
            else:
                # MACE-MP models
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

        # Classical potentials
        elif model_name in ["soft-sphere", "lennard-jones", "morse"]:
            if model_name == "soft-sphere":
                from torch_sim.models.soft_sphere import SoftSphereModel

                return SoftSphereModel(device=device, dtype=dtype)
            elif model_name == "lennard-jones":
                from torch_sim.models.lennard_jones import LennardJonesModel

                return LennardJonesModel(
                    sigma=2.0,  # Å, interaction distance
                    epsilon=0.1,  # eV, interaction strength
                    device=device,
                    dtype=dtype,
                )
            elif model_name == "morse":
                from torch_sim.models.morse import MorseModel

                return MorseModel(device=device, dtype=dtype)

        # FairChem models (example - would need actual implementation)
        elif model_name.startswith("fairchem"):
            raise NotImplementedError(
                f"FairChem models not yet implemented: {model_name}"
            )

        # SevenNet models (example - would need actual implementation)
        elif model_name.startswith("sevennet"):
            raise NotImplementedError(
                f"SevenNet models not yet implemented: {model_name}"
            )

        # ORB models (example - would need actual implementation)
        elif model_name.startswith("orb"):
            raise NotImplementedError(f"ORB models not yet implemented: {model_name}")

        else:
            raise ValueError(
                f"Unknown model: {model_name}. Supported models: mace-*, mace-off-*, soft-sphere, lennard-jones, morse"
            )

    torch_sim_model = load_torch_sim_model(model, device, dtype)

    # Convert dicts back to atoms - autobatcher handles all size optimization
    all_atoms = []
    for atoms_dict in atoms_dicts:
        atoms_dict_copy = atoms_dict.copy()
        original_index = atoms_dict_copy.pop("_index", None)
        atoms = ase.Atoms.fromdict(atoms_dict_copy)
        atoms.info["_original_index"] = original_index
        all_atoms.append(atoms)

    # Show structure statistics for debugging
    atom_counts = [len(atoms) for atoms in all_atoms]
    print(f"📊 Processing {len(all_atoms)} structures:")
    print(f"  - Atom count range: {min(atom_counts)} - {max(atom_counts)}")
    print(f"  - Average atoms per structure: {np.mean(atom_counts):.1f}")

    def generate_force_convergence_fn(force_tol=0.05):
        """Generate a convergence function based on force tolerance."""

        def convergence_fn(state):
            forces = state.force
            max_force = torch.max(torch.norm(forces, dim=-1))
            return max_force < force_tol

        return convergence_fn

    try:
        # Use torch-sim's autobatcher - it handles all memory management and batching automatically
        print("🚀 Running torch-sim batch optimization with autobatcher...")

        # Initialize state for all structures at once
        initial_state = ts.initialize_state(all_atoms, device=device, dtype=dtype)

        # Optimize entire batch using autobatcher
        relaxed_state = ts.optimize(
            system=initial_state,
            model=torch_sim_model,
            optimizer=ts.unit_cell_fire,
            max_steps=200,
            convergence_fn=generate_force_convergence_fn(force_tol=0.05),
            autobatcher=True,  # This handles all memory management automatically
        )

        # Extract results from batch - autobatcher preserves original ordering
        relaxed_atoms_list = relaxed_state.to_atoms()
        final_energies = relaxed_state.energy.cpu().tolist()

        print("✅ Autobatch optimization successful!")
        print(f"  - Average energy: {np.mean(final_energies):.3f} eV")
        print(
            f"  - Energy range: {min(final_energies):.3f} to {max(final_energies):.3f} eV"
        )

        # Process results
        all_results = []
        for atoms, energy in zip(relaxed_atoms_list, final_energies):
            result_dict = atoms.todict()

            # Convert numpy arrays to lists for serialization
            for k, v in result_dict.items():
                if isinstance(v, np.ndarray):
                    result_dict[k] = v.tolist()

            # Store final energy in info
            if "info" not in result_dict:
                result_dict["info"] = {}
            result_dict["info"]["final_energy"] = energy

            # Restore original index for sorting
            if "_original_index" in atoms.info:
                result_dict["_original_index"] = atoms.info["_original_index"]

            all_results.append(result_dict)

    except Exception as e:
        print(f"⚠️  Autobatch optimization failed: {e}")
        print("🔄 Falling back to individual processing...")

        # Fallback: process structures individually
        all_results = []
        for i, atoms in enumerate(all_atoms):
            try:
                final_state = ts.optimize(
                    system=atoms,
                    model=torch_sim_model,
                    optimizer=ts.unit_cell_fire,
                    max_steps=100,
                )

                result_atoms = final_state.to_atoms()
                if isinstance(result_atoms, list):
                    result_atoms = result_atoms[0]

                result_dict = result_atoms.todict()

                # Convert numpy arrays to lists for serialization
                for k, v in result_dict.items():
                    if isinstance(v, np.ndarray):
                        result_dict[k] = v.tolist()

                # Store energy if available
                try:
                    energy = final_state.energy.cpu().item()
                    if "info" not in result_dict:
                        result_dict["info"] = {}
                    result_dict["info"]["final_energy"] = energy
                except Exception:
                    pass

                # Restore original index
                if "_original_index" in atoms.info:
                    result_dict["_original_index"] = atoms.info["_original_index"]

                all_results.append(result_dict)

            except Exception as e2:
                print(f"  ⚠️  Individual processing failed for structure {i+1}: {e2}")
                # Use original structure as final fallback
                result_dict = atoms.todict()
                for k, v in result_dict.items():
                    if isinstance(v, np.ndarray):
                        result_dict[k] = v.tolist()
                if "_original_index" in atoms.info:
                    result_dict["_original_index"] = atoms.info["_original_index"]
                all_results.append(result_dict)

    # Sort results back to original order
    all_results.sort(key=lambda x: x.get("_original_index", 0))

    # Remove the temporary index field
    for result in all_results:
        result.pop("_original_index", None)

    print(f"✅ Batch relaxation complete! Processed {len(all_results)} structures")
    return all_results
