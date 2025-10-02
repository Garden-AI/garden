"""
Standalone batch relaxation function for HPC execution via globus-compute.

This module provides a self-contained batch_relax function that can be executed
on remote HPC systems through globus-compute executors, independent of the
MLIPGarden class.
"""


def batch_relax(
    xyz_file_path,
    model: str = "mace",
    max_batch_size: int = 200,
    relaxation_options=None,
    dtype_str: str = "float64",
) -> str:
    """
    Standalone batch relaxation function that processes structures in small batches.

    This function is designed to be executed on remote HPC systems via globus-compute
    executors, independent of the MLIPGarden class. All imports and helper functions
    are contained within this function for remote execution compatibility.

    Args:
        xyz_file_path: Path to XYZ file containing structures
        model: Model to use for relaxation (any torch-sim supported model)
        max_batch_size: Maximum structures per batch (used as fallback)
        relaxation_options: Additional options passed through to the relaxation function
        dtype_str: Data type string ('float64' or 'float32')

    Returns:
        String content of the results XYZ file
    """
    # All imports must be inside the function for remote execution
    import os
    import uuid
    from io import StringIO
    from pathlib import Path

    # Fix NUMEXPR threading issue early
    os.environ["NUMEXPR_MAX_THREADS"] = "256"

    import torch
    import torch_sim as ts
    from ase.io import read, write
    from torch_sim.runners import generate_force_convergence_fn

    def validate_relaxation_params(relax_params, frechet_supported: bool = True):
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

    # Read file content
    xyz_path = Path(xyz_file_path)
    with open(xyz_path, "r") as f:
        xyz_content = f.read()

    # Validate relaxation parameters
    validated_params, validation_error = validate_relaxation_params(relaxation_options)
    if validation_error:
        raise ValueError(f"Invalid relaxation parameters: {validation_error}")

    # Parse xyz content using StringIO
    string_file = StringIO(xyz_content)
    all_atoms = read(string_file, index=":", format="extxyz")
    num_structures = len(all_atoms)

    # Create results file path
    suffix = "relaxed"
    results_filename = f"{xyz_path.stem}_{suffix}_{uuid.uuid4().hex[:8]}.xyz"
    results_path = xyz_path.parent / results_filename

    print(f"📝 Results will be saved to: {results_path}")
    print(f"🚀 Starting batch relaxation of {num_structures} structures...")

    # Set up device and resolve dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype_str == "float64":
        torch_dtype = torch.float64
    elif dtype_str == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype string: {dtype_str}")

    # Extract relaxation parameters
    optimizer_type = validated_params.get("optimizer_type", "frechet_cell_fire")
    md_flavor = validated_params.get("md_flavor", "ase_fire")
    fmax = validated_params.get("fmax", 0.05)
    max_steps = validated_params.get("max_steps", 500)

    if optimizer_type == "frechet_cell_fire":
        optimizer_builder = ts.frechet_cell_fire
        optimize_kwargs = {
            "convergence_fn": generate_force_convergence_fn(force_tol=fmax)
        }
        # Add frechet-specific options
        frechet_kwargs = {}
        if "hydrostatic_strain" in validated_params:
            frechet_kwargs["hydrostatic_strain"] = validated_params[
                "hydrostatic_strain"
            ]
        if "constant_volume" in validated_params:
            frechet_kwargs["constant_volume"] = validated_params["constant_volume"]
        if "scalar_pressure" in validated_params:
            frechet_kwargs["scalar_pressure"] = validated_params["scalar_pressure"]
    elif optimizer_type == "fire":
        optimizer_builder = ts.optimizers.fire
        optimize_kwargs = {}
        frechet_kwargs = {}
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")

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

            # Run relaxation computation
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
        f"\n✅ Relaxation complete! Returning results for {num_structures} structures"
    )
    return results_content
