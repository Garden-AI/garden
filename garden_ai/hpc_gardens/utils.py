from pathlib import Path
import time
import base64

from globus_compute_sdk.sdk.asynchronous.compute_future import ComputeFuture


FIVE_MB = 5 * 1000 * 1000


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
            f"File size ({file_size / (1000*1000):.1f} MB) exceeds maximum allowed size "
            f"({max_size_bytes / (1000*1000):.1f} MB). Please use a smaller file."
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
    conda_env = kwargs.pop("conda_env", "torch-sim-edith-mace")
    env_path_str = f"/home/hholb/.conda/envs/{conda_env}"

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
            env_path_str,
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


def wait_for_task_id(future: ComputeFuture, timeout: int = 60) -> str:
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
