"""Test Matbench Discovery benchmark locally on Mac.

This script tests the benchmark implementation locally. Note that MPS (Apple Silicon
GPU) is not compatible with MACE model checkpoints which use float64, so this runs
on CPU. This is still useful for verifying the workflow works before using Anvil.
"""

from garden_ai.benchmarks import MatbenchDiscovery

print("=" * 80)
print("Matbench Discovery Local Test")
print("=" * 80)

# Run benchmark locally with MPS acceleration
with MatbenchDiscovery() as bench:
    task = bench.tasks.IS2RE

    print("\nRunning local benchmark...")
    print("Note: Using CPU because MACE model checkpoints use float64,")
    print("which is not supported by MPS. This is still useful for testing")
    print("the workflow before running on Anvil with CUDA.\n")

    result = task.local(
        model_package="mace-torch",
        model_factory="mace_mp",
        model_kwargs={
            "model": "medium",
            "device": "cpu",  # MPS doesn't support float64 used by MACE checkpoints
            "default_dtype": "float32",
        },
        num_structures=10,  # Small test to verify workflow
        use_multi_gpu=False,
    )

    # Calculate metrics
    metrics = task.calculate_metrics(result)

    # Display results
    print("\nResults:")
    print("=" * 80)
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("=" * 80)

    print("\nLocal test complete!")
    print("If this works, you can proceed with confidence to run on Anvil.")
