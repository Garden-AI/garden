# Matbench Discovery Benchmark Adapter

Minimal viable implementation for running [Matbench Discovery](https://matbench-discovery.materialsproject.org/) benchmarks on remote HPC systems via Globus Compute.

## Overview

This adapter enables Garden AI users to benchmark their materials models against the Matbench Discovery test suite without manually managing HPC jobs, environment setup, or data transfers.

### Current Status: MVP

**Implemented:**
- ✅ IS2RE (Initial Structure to Relaxed Energy) task
- ✅ Remote environment setup with UV
- ✅ Automatic dependency installation
- ✅ Basic metric calculation
- ✅ Multi-GPU parallelization (automatic GPU detection and work distribution)

**Future Work:**
- ⏳ Additional tasks (RS2RE, S2EFS, thermal conductivity)
- ⏳ Globus Transfer for model weights and large datasets
- ⏳ Checkpointing and failure recovery
- ⏳ Full metric calculation against DFT ground truth
- ⏳ Backend integration for result publishing

## Architecture

```
User's Machine                    Remote HPC Endpoint
├─ MatbenchDiscovery             ├─ Clone matbench-discovery repo
│  ├─ tasks.IS2RE                │  ├─ Set up UV virtual environment
│  └─ Globus Compute Executor ───┼─>├─ Install dependencies
                                 │  │  ├─ matbench-discovery
                                 │  │  └─ model package (e.g., mace-torch)
                                 │  ├─ Load test structures via DataFiles
                                 │  ├─ Run structure relaxations
                                 │  ├─ Calculate metrics
                                 │  └─ Return results
```

## File Structure

```
matbench_discovery/
├── __init__.py         # Main adapter class (MatbenchDiscovery)
├── tasks.py            # Task implementations (IS2RETask)
├── remote_runner.py    # Remote execution functions
├── enums.py            # Task enumerations
├── example.py          # Usage example
└── README.md           # This file
```

## Usage

### Basic Example

```python
from garden_ai.benchmarks import MatbenchDiscovery
from my_model import MyModel

# Configure endpoint
endpoint_id = "your-endpoint-uuid"
endpoint_config = {
    "account": "project-account",
    "partition": "gpu-debug",
    "scheduler_options": "#SBATCH --gpus-per-node=1"
}

# Run benchmark
with MatbenchDiscovery(endpoint_id, endpoint_config) as bench:
    model = MyModel()
    task = bench.tasks.IS2RE

    # Submit job (returns immediately)
    future = task.submit(model, num_structures=100)

    # Wait for completion
    results = future.result()

    # Calculate metrics
    metrics = task.calculate_metrics(results)
    print(metrics)
```

### Multi-GPU Parallelization

The adapter automatically detects and uses all available GPUs on the compute node for parallel processing. This significantly improves throughput for large-scale benchmarks.

**Example: 4-GPU Configuration on Anvil**

```python
from garden_ai.benchmarks import MatbenchDiscovery

endpoint_id = "your-endpoint-uuid"
endpoint_config = {
    "account": "your-account",
    "qos": "gpu",
    "partition": "gpu",
    "scheduler_options": "#SBATCH --gpus-per-node=4\n#SBATCH --time=4:00:00\n#SBATCH --mem=64G",
    "worker_init": "pip install --user uv",
}

with MatbenchDiscovery(endpoint_id, endpoint_config) as bench:
    task = bench.tasks.IS2RE

    # Multi-GPU is enabled by default
    future = task.submit(
        model_package="mace-torch",
        model_factory="mace_mp",
        model_kwargs={"model": "medium", "device": "cuda"},
        num_structures=1000,
        use_multi_gpu=True,  # Default: True
    )

    results = future.result()
    metrics = task.calculate_metrics(results)
```

**How it works:**
1. Automatically detects available GPUs using `torch.cuda.device_count()`
2. Splits structures into equal batches (one per GPU)
3. Processes batches in parallel using multiprocessing
4. Aggregates results from all workers

**Performance expectations:**
- **Single GPU**: ~10-20 structures/hour (baseline)
- **4 GPUs**: ~3-4x speedup (~40-80 structures/hour)
- Actual performance depends on model complexity and structure size

**Disabling multi-GPU:**
```python
future = task.submit(
    model_package="mace-torch",
    model_factory="mace_mp",
    model_kwargs={"model": "medium", "device": "cuda"},
    num_structures=100,
    use_multi_gpu=False,  # Use single GPU/CPU
)
```

### Scaling Guide

**Recommended test progression:**

1. **Small test (10-100 structures)**: Verify setup and model compatibility
   - Partition: `gpu-debug`
   - Time: 30 minutes
   - GPUs: 1-4

2. **Medium test (1000 structures)**: Test multi-GPU parallelization
   - Partition: `gpu`
   - Time: 4 hours
   - GPUs: 4
   - Expected throughput: ~250-300 structures/hour with 4 GPUs

3. **Full dataset (~257k structures)**: Production run
   - Partition: `gpu`
   - Time: 48+ hours
   - GPUs: 4
   - Consider implementing checkpointing for runs >24 hours

### Model Requirements

For the MVP, models must:

1. **Be pip-installable** (or provide package name)
2. **Implement ASE calculator interface** (or be convertible to one)
3. **Have a checkpoint file** (optional, can be None for models with default weights)

Example model:

```python
class MyModel:
    def __init__(self):
        self.checkpoint_path = "/path/to/checkpoint.pt"

    # ASE calculator interface
    def calculate(self, atoms, properties, system_changes):
        # Calculate energy, forces, stress
        ...
```

### Workflow Details

When you call `task.submit(model)`:

1. **Model introspection**: Extracts model class name, module, and checkpoint path
2. **Remote submission**: Sends job to Globus Compute endpoint
3. **Environment setup** (on remote):
   - Clones matbench-discovery repository
   - Creates Python 3.11 virtual environment with UV
   - Installs matbench-discovery package
   - Installs model package (e.g., `pip install mace-torch`)
4. **Benchmark execution**:
   - Loads test structures using `DataFiles.wbm_initial_structures`
   - Instantiates model and loads checkpoint
   - Runs geometry optimizations (ASE FIRE optimizer)
   - Collects results
5. **Result return**: Returns energies, convergence stats, and failures

## Configuration Options

### MatbenchDiscovery

```python
MatbenchDiscovery(
    endpoint_id="uuid",           # Required: Globus Compute endpoint
    user_endpoint_config=dict,     # Optional: HPC scheduler config
    repo_ref="main",               # Optional: Git ref to use
    model_package="mace-torch"     # Optional: Default model package
)
```

### IS2RETask.submit()

```python
task.submit(
    model,                         # Required: Model instance
    num_structures=100,            # Optional: Number of structures to test
    model_package="mace-torch",    # Optional: Override default package
    use_multi_gpu=True,            # Optional: Enable multi-GPU (default: True)
)
```

## Design Decisions

### Why UV?
- Fast, deterministic installs
- Handles both `pyproject.toml` and `requirements.txt`
- Built-in venv creation with specific Python versions

### Why DataFiles auto-download?
- Avoids manual Globus Transfer setup for MVP
- Matbench's DataFiles handles caching automatically
- Can optimize with explicit transfer later

### Why ASE calculator interface?
- Standard in materials modeling community
- Most interatomic potentials support it (MACE, M3GNet, CHGNet, etc.)
- Simple adaptation layer if needed

### Why multiprocessing for multi-GPU?
- Simple and effective for within-node parallelization
- Avoids CUDA initialization issues with fork
- Each GPU gets isolated process with dedicated memory
- Easy to debug and monitor per-GPU progress

## Limitations

1. **No weight transfer**: Model checkpoints must be accessible from remote (URL or shared filesystem)
2. **Basic metrics**: Only reports convergence stats, not comparison to DFT ground truth
3. **IS2RE only**: Other tasks not yet implemented
4. **No checkpointing**: If job fails, must restart from scratch (recommended for runs >24 hours)
5. **No result publishing**: Backend integration not yet implemented
6. **Single-node parallelization**: Multi-GPU works within a node; SLURM array jobs for multi-node not yet implemented

## Next Steps

To generalize beyond Matbench:

1. **Extract base classes**: `BenchmarkAdapter`, `BenchmarkTask`, `RemoteRunner`
2. **Add data staging**: Implement Globus Transfer for weights/datasets
3. **Define model interface**: Standard protocol for model serialization
4. **Add checkpointing**: Save intermediate results for failure recovery
5. **Implement batching**: Distribute work across SLURM array jobs

## Testing

```bash
# Install dependencies
cd garden_ai/benchmarks/matbench_discovery
pip install -e .

# Update example.py with your endpoint details
vim example.py

# Run example
python example.py
```

## References

- [Matbench Discovery](https://matbench-discovery.materialsproject.org/)
- [Matbench Discovery GitHub](https://github.com/janosh/matbench-discovery)
- [Globus Compute](https://globus-compute.readthedocs.io/)
- [ASE Calculator Interface](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html)
