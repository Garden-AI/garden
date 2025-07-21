import pickle
import base64
from typing import Any
from pathlib import Path
import inspect
from globus_compute_sdk import Executor
from globus_compute_sdk.serialize import ComputeSerializer, CombinedCode

from garden_ai.hpc_gardens.utils import generate_xyz_str_chunks

EDITH_EP_ID = "a01b9350-e57d-4c8e-ad95-b4cb3c4cd1bb"

DEFAULT_CONFIG = {
    "worker_init": """
        module load openmpi
        export PATH=$PATH:/usr/sbin
    """,
}


class EdithExecutor(Executor):
    def __init__(
        self,
        *args,
        endpoint_id=EDITH_EP_ID,
        user_endpoint_config=DEFAULT_CONFIG,
        **kwargs,
    ):
        super().__init__(
            *args,
            endpoint_id=endpoint_id,
            user_endpoint_config=user_endpoint_config,
            **kwargs,
        )
        self.serializer = ComputeSerializer(strategy_code=CombinedCode())

    def submit(self, func, *args, **kwargs) -> Any:
        """
        Execute a function on a Globus Compute endpoint using subprocess with conda environment.

        Args:
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            the globus-compute future
        """
        # submit the function the to gcmu exector
        func_source = inspect.getsource(func)
        fut = super().submit(_subproc_wrapper, func_source, *args, **kwargs)
        return fut

    def _parse_results(self, result):
        # If we got raw data back, try to unpickle it locally
        if "raw_data" in result:
            try:
                actual_result = self.decode_result_data(result["raw_data"])
                return actual_result
            except Exception as e:
                return {
                    "error": f"Failed to decode result locally: {e}",
                    "raw_data": result["raw_data"],
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", ""),
                }
        else:
            # Return the result as-is if no raw_data key
            return result

    def decode_result_data(self, raw_data: str):
        """
        Decode base64 pickled result data when ASE is available.

        Args:
            raw_data: Base64 encoded pickled data

        Returns:
            Unpickled result object
        """
        return pickle.loads(base64.b64decode(raw_data))


def send_xyz_to_endpoint(xyx_file: str | Path, endpoint_id=EDITH_EP_ID):
    chunk_files = []
    ex = EdithExecutor(endpoint_id=endpoint_id)
    for chunk in generate_xyz_str_chunks(xyx_file):
        f = ex.submit(_send_chunk_to_endpoint, chunk)
        result = f.result()
        filename = ex._parse_results(result)

        # Check if we got an error instead of a filename
        if isinstance(filename, dict) and "error" in filename:
            return filename

        chunk_files.append(filename)

    f = ex.submit(_collate_file_chunks, xyx_file, chunk_files)
    result = f.result()
    return ex._parse_results(result)


def _subproc_wrapper(func_source, *args, **kwargs):
    """Wrapper around a function to execute in a subprorcess using a conda env on the remote endpoint.

    This is designed to be serializable by globus-compute and get around python version
    mismatches and import errors that arise when the globus-compute endpoint is using
    a different python version and environment than the caller.
    """
    import subprocess
    import pickle
    import base64
    import tempfile
    import os
    import re

    # Extract function name from the source
    func_name_match = re.search(r"def\s+(\w+)\s*\(", func_source)
    if not func_name_match:
        return {"error": "Could not extract function name from source"}

    func_name = func_name_match.group(1)

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
            "/home/hholb/.conda/envs/torch-sim-edith",
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


def _send_chunk_to_endpoint(chunk: str) -> str:
    import tempfile

    chunk_file = tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False)
    chunk_file.write(chunk)
    chunk_file.close()
    return chunk_file.name


def _collate_file_chunks(master_file: str, chunk_file_names: list[str]) -> str:
    from pathlib import Path

    master_file = Path.home().joinpath(master_file)
    with open(master_file, "a") as f:
        for name in chunk_file_names:
            file = Path(name)
            assert file.exists()
            with open(file, "r") as chunk_f:
                f.writelines(chunk_f.readlines())
    return str(master_file)
