"""Generic remote execution utility for benchmarks.

This module contains the `run_remote_benchmark` function which is designed to be
serialized and executed on Globus Compute endpoints. It handles the boilerplate
of setting up a Python environment, installing dependencies, and running a
provided benchmark script.
"""


def run_remote_benchmark(
    script_content: str,
    dependencies: list[str],
    config: dict,
    checkpoint_name: str | None = None,
    checkpoint_path: str | None = None,
) -> dict:
    """Run a generic benchmark script on a remote Globus Compute endpoint.

    This function:
    1. Creates a temporary working directory.
    2. Sets up a Python environment using `uv`.
    3. Installs the specified dependencies.
    4. Writes the `script_content` to a file.
    5. Writes the `config` to a JSON file.
    6. Executes the script in the environment.
    7. Returns the results from `results.json`.

    Args:
        script_content: The full Python script to execute.
        dependencies: List of Python packages to install (e.g. ["numpy", "torch"]).
        config: Dictionary of configuration parameters to pass to the script.
                Written to `config.json`.
        checkpoint_name: Name of the checkpoint file (e.g. "checkpoint_123.json").
                         Saved to ~/.garden/benchmarks/.
        checkpoint_path: Optional path to an existing checkpoint file to resume from.
                         If provided, this path is used directly.

    Returns:
        The content of `results.json` produced by the script.
    """
    # All imports must be inside the function for serialization
    import json
    import logging
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        force=True,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    logger = logging.getLogger(__name__)

    # Create isolated working directory
    work_dir = Path(tempfile.mkdtemp(prefix="garden_benchmark_"))

    try:
        # ----------------------------------------------------------------------
        # 1. ENVIRONMENT SETUP
        # ----------------------------------------------------------------------
        logger.info("Step 1/4: Setting up environment...")

        # Find UV binary
        try:
            uv_bin = subprocess.check_output(
                [sys.executable, "-c", "import uv; print(uv.find_uv_bin())"], text=True
            ).strip()
        except subprocess.CalledProcessError:
            import shutil

            uv_bin = shutil.which("uv")
            if not uv_bin:
                raise RuntimeError("Could not find uv binary. Please install uv.")

        # Create UV virtual environment
        subprocess.run(
            [uv_bin, "venv", "--python", "3.11"],
            cwd=work_dir,
            check=True,
            capture_output=True,
        )

        venv_python = work_dir / ".venv/bin/python"
        if not venv_python.exists():
            venv_python = work_dir / ".venv/Scripts/python.exe"  # Windows fallback

        if not venv_python.exists():
            raise RuntimeError(f"Virtual environment python not found at {venv_python}")

        # Install dependencies
        logger.info(f"Installing dependencies: {dependencies}")
        # Install in one go for better resolution
        cmd = [uv_bin, "pip", "install", "--python", str(venv_python)] + dependencies
        subprocess.run(
            cmd,
            cwd=work_dir,
            check=True,
        )

        # Set SSL cert file for HPC if needed
        env = dict(os.environ)

        # Propagate common useful env vars if present
        for key in ["MBD_AUTO_DOWNLOAD_FILES", "HF_TOKEN", "WANDB_API_KEY"]:
            if key in os.environ:
                env[key] = os.environ[key]

        try:
            certifi_path = subprocess.check_output(
                [str(venv_python), "-c", "import certifi; print(certifi.where())"],
                text=True,
            ).strip()
            env["SSL_CERT_FILE"] = certifi_path
        except Exception as e:
            logger.warning(f"Failed to set SSL_CERT_FILE: {e}")

        # ----------------------------------------------------------------------
        # 2. PREPARE BENCHMARK SCRIPT
        # ----------------------------------------------------------------------
        logger.info("Step 2/4: Preparing benchmark script...")

        # Write runner script
        runner_path = work_dir / "benchmark_runner.py"
        runner_path.write_text(script_content)

        # Determine checkpoint path
        if checkpoint_path:
            # User provided a specific path to resume from
            final_checkpoint_path = checkpoint_path
        elif checkpoint_name:
            # Use persistent location in user home
            checkpoint_dir = Path.home() / ".garden" / "benchmarks"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            final_checkpoint_path = str(checkpoint_dir / checkpoint_name)
        else:
            # Fallback to tmp dir if no name provided (legacy behavior)
            final_checkpoint_path = str(work_dir / "checkpoint.json")

        config["checkpoint_path"] = final_checkpoint_path

        # Log checkpoint path prominently for user reference
        print(f"{'=' * 80}")
        print(f"Checkpoint will be saved to: {final_checkpoint_path}")
        print("To resume this job if it fails, use:")
        print(f'  checkpoint_path="{final_checkpoint_path}"')
        print(f"{'=' * 80}")

        # Write config
        config_path = work_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # ----------------------------------------------------------------------
        # 3. EXECUTE BENCHMARK
        # ----------------------------------------------------------------------
        logger.info("Step 3/4: Executing benchmark...")

        # Run the runner script inside the venv
        # DO NOT capture output - let it stream to stdout/stderr in real-time
        # so we can see errors immediately
        proc = subprocess.run(
            [str(venv_python), str(runner_path), str(config_path)],
            cwd=work_dir,
            env=env,
            check=False,  # Don't raise immediately, we'll check returncode
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"Benchmark runner failed with return code {proc.returncode}."
            )

        # ----------------------------------------------------------------------
        # 4. COLLECT RESULTS
        # ----------------------------------------------------------------------
        logger.info("Step 4/4: Collecting results...")

        results_path = work_dir / "results.json"
        if not results_path.exists():
            raise RuntimeError(
                "Results file not found - benchmark may have crashed silently"
            )

        with open(results_path) as f:
            results = json.load(f)

        logger.info("Benchmark completed successfully.")
        return results

    finally:
        # Cleanup working directory
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)
