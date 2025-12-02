import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class RemoteBenchmarkRunner:
    """
    Handles the setup and execution of benchmarks on remote Globus Compute endpoints.

    This class manages:
    1. Creating an isolated working directory
    2. Setting up a Python environment using `uv`
    3. Installing dependencies
    4. Executing the benchmark script
    5. Collecting results
    """

    def __init__(self, work_dir_prefix: str = "garden_benchmark_"):
        self.work_dir = Path(tempfile.mkdtemp(prefix=work_dir_prefix))
        self.uv_bin = None
        self.venv_python = None
        self.env = dict(os.environ)

        # Configure logging if not already configured
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                stream=sys.stdout,
                force=True,
                format="%(asctime)s [%(levelname)s] %(message)s",
            )

    def setup_environment(self, python_version: str = "3.11"):
        """Find uv and create virtual environment."""
        logger.info("Setting up environment...")

        # Find UV binary
        try:
            self.uv_bin = subprocess.check_output(
                [sys.executable, "-c", "import uv; print(uv.find_uv_bin())"], text=True
            ).strip()
        except subprocess.CalledProcessError:
            import shutil

            self.uv_bin = shutil.which("uv")
            if not self.uv_bin:
                raise RuntimeError("Could not find uv binary. Please install uv.")

        # Create UV virtual environment
        subprocess.run(
            [self.uv_bin, "venv", "--python", python_version],
            cwd=self.work_dir,
            check=True,
            capture_output=True,
        )

        self.venv_python = self.work_dir / ".venv/bin/python"
        if not self.venv_python.exists():
            self.venv_python = (
                self.work_dir / ".venv/Scripts/python.exe"
            )  # Windows fallback

        if not self.venv_python.exists():
            raise RuntimeError(
                f"Virtual environment python not found at {self.venv_python}"
            )

        # Set SSL cert file for HPC if needed
        self._setup_ssl_cert()

    def _setup_ssl_cert(self):
        """Set SSL_CERT_FILE environment variable if certifi is available."""
        try:
            certifi_path = subprocess.check_output(
                [str(self.venv_python), "-c", "import certifi; print(certifi.where())"],
                text=True,
            ).strip()
            self.env["SSL_CERT_FILE"] = certifi_path
        except Exception as e:
            logger.warning(f"Failed to set SSL_CERT_FILE: {e}")

    def install_dependencies(self, packages: List[str]):
        """Install Python packages into the virtual environment."""
        logger.info(f"Installing dependencies: {packages}")
        if not self.uv_bin or not self.venv_python:
            raise RuntimeError("Environment not setup. Call setup_environment() first.")

        cmd = [
            self.uv_bin,
            "pip",
            "install",
            "--python",
            str(self.venv_python),
        ] + packages

        subprocess.run(cmd, cwd=self.work_dir, check=True)

    def run_benchmark(
        self,
        script_content: str,
        config: Dict[str, Any],
        script_name: str = "benchmark_runner.py",
    ) -> Dict[str, Any]:
        """
        Execute the benchmark script.

        Args:
            script_content: The Python script to run.
            config: Configuration dictionary to pass to the script (saved as config.json).
            script_name: Filename for the script.

        Returns:
            Dictionary containing the results loaded from results.json.
        """
        if not self.venv_python:
            raise RuntimeError("Environment not setup. Call setup_environment() first.")

        logger.info("Preparing benchmark script...")

        # Write runner script
        runner_path = self.work_dir / script_name
        runner_path.write_text(script_content)

        # Write config
        config_path = self.work_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Executing benchmark...")

        # Run the runner script inside the venv
        proc = subprocess.run(
            [str(self.venv_python), str(runner_path), str(config_path)],
            cwd=self.work_dir,
            env=self.env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=False,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"Benchmark runner failed with return code {proc.returncode}"
            )

        logger.info("Collecting results...")
        results_path = self.work_dir / "results.json"
        if not results_path.exists():
            raise RuntimeError(
                "Results file not found - benchmark may have crashed silently"
            )

        with open(results_path) as f:
            results = json.load(f)

        logger.info("Benchmark completed successfully.")
        return results

    def cleanup(self):
        """Remove the working directory."""
        import shutil

        shutil.rmtree(self.work_dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
