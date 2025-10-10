import base64
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from globus_compute_sdk import Client as GlobusComputeClient
from groundhog_hpc.future import GroundhogFuture

FIVE_MB = 5 * 1000 * 1000


@dataclass
class JobStatus:
    """Status information for an HPC batch job."""

    status: str  # "pending" | "running" | "completed" | "failed" | "unknown"
    results_available: bool = False
    error: str | None = None
    stdout: str = ""
    stderr: str = ""


def check_file_size_and_read(xyz_file, max_size_bytes=FIVE_MB):
    """Check if file is under size limit and return its contents as string.

    Args:
        xyz_file: Path to XYZ file
        max_size_bytes: Maximum allowed file size in bytes

    Returns:
        String contents of the file

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file exceeds size limit
    """
    xyz_path = Path(xyz_file)
    if not xyz_path.exists():
        raise FileNotFoundError(f"XYZ file not found: {xyz_file}")

    file_size = xyz_path.stat().st_size
    if file_size > max_size_bytes:
        raise ValueError(
            f"File size ({file_size / (1000 * 1000):.1f} MB) exceeds maximum allowed size "
            f"({max_size_bytes / (1000 * 1000):.1f} MB). Please use a smaller file."
        )

    with open(xyz_path, "r") as f:
        return f.read()


def subproc_wrapper(func_source, *args, **kwargs):
    """Wrapper around a function to execute in a subprorcess using a conda env on the remote endpoint.

    This is designed to be serializable by globus-compute and get around python version
    mismatches and import errors that arise when the globus-compute endpoint is using
    a different python version and environment than the caller.
    """
    import base64
    import os
    import pickle
    import re
    import subprocess
    import tempfile

    # Extract function name from the source
    func_name_match = re.search(r"def\s+(\w+)\s*\(", func_source)
    if not func_name_match:
        return {"error": "Could not extract function name from source"}

    func_name = func_name_match.group(1)
    conda_env_path = kwargs.pop("conda_env_path", None)

    if conda_env_path is None:
        return {"error": "conda_env_path is required but was not provided"}

    # Function data to execute
    func_data = {
        "source": func_source,
        "name": func_name,
        "args": args,
        "kwargs": kwargs,
    }

    # Encode function data
    encoded_data = base64.b64encode(pickle.dumps(func_data)).decode()

    # Python script to run in conda environment
    script = f"""import pickle
import base64

# Decode function data
func_data = pickle.loads(base64.b64decode("{encoded_data}"))

# Execute function source to define it
exec(func_data["source"])

# Get function object and execute it
func_obj = locals()[func_data["name"]]
result = func_obj(*func_data["args"], **func_data["kwargs"])

# Serialize and print result for capture
result_data = base64.b64encode(pickle.dumps(result)).decode()
print("RESULT_DATA:", result_data)
"""

    # Write script to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        # Run in template script in conda env
        cmd = [
            "conda",
            "run",
            "-p",
            conda_env_path,
            "python",
            script_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        # Clean up temporary file
        os.unlink(script_path)

    if result.returncode != 0:
        return {"error": result.stderr, "stdout": result.stdout}

    # Extract result from subprocess output
    result_data = None

    for line in result.stdout.split("\n"):
        if line.startswith("RESULT_DATA: "):
            result_data = line[13:].strip()
            break

    if result_data is not None:
        return {
            "raw_data": result_data,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    return {
        "error": "No result found",
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def wait_for_task_id(future: GroundhogFuture, timeout: int = 60) -> str:
    """Waits for a globus-compute task ID to become available, with a timeout."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        task_id = future.task_id
        if task_id:
            return task_id
        time.sleep(0.5)
    raise TimeoutError(
        f"Could not get task ID from globus-compute within {timeout} seconds."
    )


def get_job_status(job_id: str, gc_client: GlobusComputeClient) -> JobStatus:
    """
    Get status information for a submitted HPC job.

    Args:
        job_id: Globus Compute task ID returned by HpcFunction.submit()
        gc_client: Globus Compute client instance

    Returns:
        JobStatus object with current status and outputs

    Example:
        >>> from globus_compute_sdk import Client as GlobusComputeClient
        >>> gc_client = GlobusComputeClient()
        >>> status = get_job_status(job_id, gc_client)
        >>> print(status.status)  # "completed"
        >>> if status.status == "completed":
        ...     results = get_results(job_id, gc_client)
    """
    task_info = gc_client.get_task(job_id)

    # Check if still pending
    if task_info.get("pending", True):
        return JobStatus(status="pending")

    # Try to get result
    try:
        job_result = gc_client.get_result(job_id)

        # Check for error in result
        if isinstance(job_result, dict) and "error" in job_result:
            return JobStatus(
                status="failed",
                error=job_result["error"],
                stdout=decode_if_base64(job_result.get("stdout", "")),
                stderr=decode_if_base64(job_result.get("stderr", "")),
            )

        # Success - results available
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
    job_id: str,
    gc_client: GlobusComputeClient,
    output_path: str | Path | None = None,
) -> Any:
    """
    Retrieve results from a completed HPC job.

    Args:
        job_id: Globus Compute task ID
        gc_client: Globus Compute client instance
        output_path: Optional local path to save results (for file-based results)

    Returns:
        Job results (type depends on the function)

    Raises:
        RuntimeError: If job is not completed or failed

    Example:
        >>> from globus_compute_sdk import Client as GlobusComputeClient
        >>> gc_client = GlobusComputeClient()
        >>> results = get_results(job_id, gc_client)
        >>> # Or save to file:
        >>> results = get_results(job_id, gc_client, output_path="./results.xyz")
    """
    # Check status first
    status_info = get_job_status(job_id, gc_client)

    if status_info.status == "pending":
        raise RuntimeError(f"Job {job_id} is still pending")
    elif status_info.status == "running":
        raise RuntimeError(f"Job {job_id} is still running")
    elif status_info.status == "failed":
        raise RuntimeError(f"Job {job_id} failed: {status_info.error}")
    elif status_info.status != "completed":
        raise RuntimeError(f"Job {job_id} has unknown status")

    if not status_info.results_available:
        raise RuntimeError(f"No results available for job {job_id}")

    # Get the actual result
    job_result = gc_client.get_result(job_id)

    # Handle encoded data if present (from subproc_wrapper)
    if isinstance(job_result, dict) and "raw_data" in job_result:
        actual_result = pickle.loads(base64.b64decode(job_result["raw_data"]))
    else:
        actual_result = job_result

    # Save to file if requested
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(actual_result, str):
            with open(output_path, "w") as f:
                f.write(actual_result)
        else:
            # For non-string results, try to pickle or stringify
            try:
                with open(output_path, "wb") as f:
                    pickle.dump(actual_result, f)
            except Exception:
                with open(output_path, "w") as f:
                    f.write(str(actual_result))

        print(f"✅ Results saved to {output_path}")

    return actual_result


def decode_if_base64(output_text: str) -> str:
    """
    Decode base64-encoded output text if it is base64, otherwise return as-is.
    Also removes RESULT_DATA lines from the output.

    Args:
        output_text: Raw output text that may be base64-encoded

    Returns:
        Decoded text with RESULT_DATA lines removed
    """
    if not output_text:
        return output_text

    # Try to decode from base64, if it fails just use original
    try:
        decoded_bytes = base64.b64decode(output_text)
        decoded_text = decoded_bytes.decode("utf-8")
        output_text = decoded_text
    except Exception:
        # If any error occurs, just use the original text
        pass

    # Filter out RESULT_DATA lines
    lines = output_text.split("\n")
    cleaned_lines = [line for line in lines if not line.startswith("RESULT_DATA:")]

    # Remove trailing empty lines
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    return "\n".join(cleaned_lines)
