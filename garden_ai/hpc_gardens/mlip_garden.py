from pathlib import Path
from typing import Any

from garden_ai.gardens import Garden
from garden_ai.hpc_executors.edith_executor import (
    EdithExecutor,
    _send_chunk_to_endpoint,
    _collate_file_chunks,
    _stream_result_chunk_from_file,
)
from garden_ai.hpc_gardens.utils import generate_xyz_str_chunks
from garden_ai.schemas.garden import GardenMetadata
from globus_compute_sdk import Client


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

    def run_relaxation(
        self, atoms_dict, model: str = "mace-mp-0", cluster_id: str | None = None
    ):
        """
        Run relaxation on-demand. This is intended for small inputs and testing.
        For large inputs use the batch_relax flow.

        Args:
            atoms: ASE Atoms object to relax
            model: Model to use for relaxation (any torch-sim supported model)
            cluster_id: HPC endpoint ID to use for computation

        Returns:
            relaxed atoms dict
        """
        ex = EdithExecutor(endpoint_id=cluster_id if cluster_id else None)

        # Submit the function
        fut = ex.submit(_run_single_relaxation, atoms_dict, model)
        raw_result = fut.result()
        decoded = ex.decode_result_data(raw_result.get("raw_data"))
        return decoded

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
        import uuid  # noqa:
        from pathlib import Path

        # Read structures locally
        xyz_path = Path(xyz_file_path)
        if not xyz_path.exists():
            raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

        ex = EdithExecutor(endpoint_id=cluster_id)
        chunks = []
        for chunk in generate_xyz_str_chunks(xyz_path):
            f = ex.submit(_send_chunk_to_endpoint, chunk)
            # wait for each chunk
            chunk_result = f.result()

            # Check if chunk creation failed
            if isinstance(chunk_result, dict) and "error" in chunk_result:
                raise RuntimeError(
                    f"Failed to send chunk to endpoint: {chunk_result['error']}"
                )

            # Extract the actual filename from the result
            if isinstance(chunk_result, dict) and "raw_data" in chunk_result:
                chunk_filename = ex.decode_result_data(chunk_result["raw_data"])
                chunks.append(chunk_filename)
            else:
                chunks.append(chunk_result)

        # Use the same filename as the input file on the remote endpoint
        master_filename = xyz_path.name
        f = ex.submit(_collate_file_chunks, master_filename, chunks)
        collate_result = f.result()

        # Check if file collation failed
        if isinstance(collate_result, dict) and "error" in collate_result:
            raise RuntimeError(
                f"Failed to collate file chunks: {collate_result['error']}"
            )

        # Extract the actual filename from the collate result
        if isinstance(collate_result, dict) and "raw_data" in collate_result:
            input_file = ex.decode_result_data(collate_result["raw_data"])
        else:
            input_file = collate_result
        future = ex.submit(_run_batch_relaxation, input_file, model, max_batch_size)
        task_id = future.task_id

        # TODO: this could be an infinite loop if we never get the task id
        while task_id is None:
            task_id = future.task_id

        return task_id

    def get_job_status(self, job_id: str) -> dict:
        """
        Get comprehensive status information for a submitted batch job.

        Args:
            job_id: Job ID returned by batch_relax

        Returns:
            Dictionary with status info:
            {
                "status": "pending" | "running" | "completed" | "failed",
                "remote_file_path": str,  # if completed successfully
                "error": str,             # if failed
                "stdout": str,            # job output
                "stderr": str             # warnings/errors
            }
        """
        client = Client()
        task_info = client.get_task(job_id)

        # Extract basic status
        if task_info.get("pending", True):
            return {
                "status": "pending",
                "remote_file_path": None,
                "error": None,
                "stdout": "",
                "stderr": "",
            }

        # Get the actual result to determine if completed or failed
        try:
            job_result = client.get_result(job_id)

            if isinstance(job_result, dict) and "error" in job_result:
                return {
                    "status": "failed",
                    "remote_file_path": None,
                    "error": job_result["error"],
                    "stdout": job_result.get("stdout", ""),
                    "stderr": job_result.get("stderr", ""),
                }

            # Successful completion - decode remote file path
            remote_file_path = None
            if isinstance(job_result, dict) and "raw_data" in job_result:
                ex = EdithExecutor()
                remote_file_path = ex.decode_result_data(job_result["raw_data"])

            return {
                "status": "completed",
                "remote_file_path": remote_file_path,
                "error": None,
                "stdout": (
                    job_result.get("stdout", "") if isinstance(job_result, dict) else ""
                ),
                "stderr": (
                    job_result.get("stderr", "") if isinstance(job_result, dict) else ""
                ),
            }

        except Exception as e:
            return {
                "status": "unknown",
                "remote_file_path": None,
                "error": f"Failed to get job status: {str(e)}",
                "stdout": "",
                "stderr": "",
            }

    def get_results(
        self, job_id: str, output_path: str | Path = None, cluster_id: str | None = None
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

        if status_info["status"] == "pending":
            raise RuntimeError(f"Job {job_id} is still pending")
        elif status_info["status"] == "running":
            raise RuntimeError(f"Job {job_id} is still running")
        elif status_info["status"] == "failed":
            raise RuntimeError(f"Job {job_id} failed: {status_info['error']}")
        elif status_info["status"] != "completed":
            raise RuntimeError(
                f"Job {job_id} has unknown status: {status_info['status']}"
            )

        remote_file_path = status_info["remote_file_path"]
        if not remote_file_path:
            raise RuntimeError(f"No results file found for completed job {job_id}")

        # Set up output path
        if output_path is None:
            output_path = Path(f"./results_{job_id}.xyz")
        else:
            output_path = Path(output_path)

        # Set up executor for streaming - use provided cluster_id or default to EDITH endpoint
        from garden_ai.hpc_executors.edith_executor import EDITH_EP_ID

        endpoint_id = cluster_id if cluster_id else EDITH_EP_ID
        ex = EdithExecutor(endpoint_id=endpoint_id)

        # Stream results in chunks
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"📥 Downloading results from remote: {remote_file_path}")
        print(f"📁 Saving to local file: {output_path}")

        chunk_index = 0
        total_chunks = None

        while True:
            # Get next chunk from remote file
            f = ex.submit(_stream_result_chunk_from_file, remote_file_path, chunk_index)
            chunk_result = f.result()

            # Handle result decoding if needed
            if isinstance(chunk_result, dict) and "raw_data" in chunk_result:
                chunk_result = ex.decode_result_data(chunk_result["raw_data"])

            # Check for errors
            if isinstance(chunk_result, dict) and "error" in chunk_result:
                raise RuntimeError(
                    f"Failed to download results: {chunk_result['error']}"
                )

            # Extract chunk data
            chunk_data = chunk_result["chunk_data"]
            chunk_index = chunk_result["chunk_index"]
            total_chunks = chunk_result["total_chunks"]
            is_complete = chunk_result["is_complete"]

            # Write chunk to local file (first chunk overwrites, rest append)
            mode = "w" if chunk_index == 0 else "a"
            with open(output_path, mode) as f:
                f.write(chunk_data)

            print(f"📦 Downloaded chunk {chunk_index + 1}/{total_chunks}")

            # Check if we're done
            if is_complete:
                break

            chunk_index += 1

        print(f"✅ Successfully downloaded {total_chunks} chunks to {output_path}")
        return output_path


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

        def convergence_fn(state, last_energy):
            forces = state.forces
            # Calculate max force per batch (not global max)
            force_norms = torch.norm(forces, dim=-1)  # Shape: [n_atoms]
            # Reshape to [n_batches, atoms_per_batch] and get max per batch
            n_batches = state.n_batches
            atoms_per_batch = force_norms.shape[0] // n_batches
            force_norms_batched = force_norms.view(n_batches, atoms_per_batch)
            max_forces_per_batch = torch.max(force_norms_batched, dim=-1)[
                0
            ]  # Shape: [n_batches]
            return max_forces_per_batch < force_tol

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

    # For single relaxation, use a simple MACE model loading
    from mace.calculators.foundations_models import mace_mp
    from torch_sim.models.mace import MaceModel

    print(f"🔧 Loading {model} model on {device}...")
    loaded_model = mace_mp(
        model="medium", return_raw_model=True, default_dtype=dtype, device=device
    )
    torch_sim_model = MaceModel(
        model=loaded_model,
        device=device,
        compute_forces=True,
        compute_stress=True,
        dtype=dtype,
    )

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


def _run_batch_relaxation(xyz_filename, model="mace-mp-0", max_batch_size=10):
    """
    Run batch relaxation on multiple structures using torch-sim's autobatcher.
    Saves results incrementally to handle 30-minute time limit gracefully.

    Args:
        xyz_filename: Path to xyz file containing structures on remote endpoint
        model: Model to use for relaxation (any torch-sim supported model)
        max_batch_size: Maximum structures to process at once (legacy parameter, autobatcher handles this)

    Returns:
        Path to results XYZ file on remote endpoint
    """
    import ase  # noqa:
    import numpy as np
    import torch
    import torch_sim as ts
    from ase.io import read, write
    from pathlib import Path
    import uuid

    # Read structures from xyz file
    print(f"📖 Reading structures from {xyz_filename}...")
    all_atoms = read(xyz_filename, index=":")

    # Create results file path
    input_path = Path(xyz_filename)
    results_filename = f"{input_path.stem}_relaxed_{uuid.uuid4().hex[:8]}.xyz"
    results_path = input_path.parent / results_filename

    print(f"📝 Results will be saved to: {results_path}")
    print(
        f"🚀 Starting batch relaxation of {len(all_atoms)} structures using torch-sim autobatcher..."
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

    # Add original index to track structure order
    for i, atoms in enumerate(all_atoms):
        atoms.info["_original_index"] = i

    # Show structure statistics for debugging
    atom_counts = [len(atoms) for atoms in all_atoms]
    print(f"📊 Processing {len(all_atoms)} structures:")
    print(f"  - Atom count range: {min(atom_counts)} - {max(atom_counts)}")
    print(f"  - Average atoms per structure: {np.mean(atom_counts):.1f}")

    def generate_force_convergence_fn(force_tol=0.05):
        """Generate a convergence function based on force tolerance."""

        def convergence_fn(state, last_energy):
            forces = state.forces
            # Calculate max force per batch (not global max)
            force_norms = torch.norm(forces, dim=-1)  # Shape: [n_atoms]
            # Reshape to [n_batches, atoms_per_batch] and get max per batch
            n_batches = state.n_batches
            atoms_per_batch = force_norms.shape[0] // n_batches
            force_norms_batched = force_norms.view(n_batches, atoms_per_batch)
            max_forces_per_batch = torch.max(force_norms_batched, dim=-1)[
                0
            ]  # Shape: [n_batches]
            return max_forces_per_batch < force_tol

        return convergence_fn

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

    # Save results incrementally to XYZ file
    print(
        f"💾 Saving {len(relaxed_atoms_list)} relaxed structures to {results_path}..."
    )
    for i, (atoms, energy) in enumerate(zip(relaxed_atoms_list, final_energies)):
        # Store energy in atoms info for XYZ comment
        atoms.info["energy"] = energy
        atoms.info["model"] = model
        atoms.info["relaxed"] = True

        # Write to XYZ file (append mode after first structure)
        write(results_path, atoms, append=(i > 0))

    print(f"✅ Saved all {len(relaxed_atoms_list)} structures to XYZ file!")
    print(f"📁 Results file: {results_path}")

    return str(results_path)
