"""Test Matbench Discovery benchmark locally."""

from garden_ai.benchmarks import MatbenchDiscovery

print("Matbench Discovery IS2RE Benchmark")
print("=" * 80)

with MatbenchDiscovery() as bench:
    task = bench.tasks.IS2RE

    # Run benchmark locally
    result = task.local(
        model_package="mace-torch",
        model_factory="mace_mp",
        model_kwargs={
            "model": "medium",
            "device": "cpu",
            "default_dtype": "float32",
        },
        num_structures=10,
    )

    # Calculate metrics
    metrics = task.calculate_metrics(result)

    # Display results
    print("\nResults:")
    print("=" * 80)
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("=" * 80)
