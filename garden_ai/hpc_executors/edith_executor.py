import pickle
import base64
import uuid
from typing import Any
from pathlib import Path
import inspect
from globus_compute_sdk import Executor
from globus_compute_sdk.serialize import ComputeSerializer, CombinedCode

EDITH_EP_ID = "a01b9350-e57d-4c8e-ad95-b4cb3c4cd1bb"


class EdithExecutor:
    def __init__(self, endpoint_id=EDITH_EP_ID):
        self.endpoint_id = endpoint_id
        self.endpoint_config = {
            # some key details about the worker init script:
            # for multiple commands it must be a multi-line string, no "command 1; command2 2;"
            # file IO seems to break it sometimes, but not always?
            "worker_init": """
                module load openmpi
                export PATH=$PATH:/usr/sbin
            """,
            # TODO: figure out a more optimal job config
            # "engine": {
            #     "provider": {
            #         "nodes_per_block": 2,
            #     },
            # }
        }
        self.executor = Executor(
            endpoint_id=self.endpoint_id, user_endpoint_config=self.endpoint_config
        )
        self.executor.serializer = ComputeSerializer(strategy_code=CombinedCode())

    def submit(self, func, *args, **kwargs) -> Any:
        """
        Execute a function on a Globus Compute endpoint using subprocess with conda environment.

        Args:
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            the task id of the function
        """
        # submit the function the to gcmu exector
        func_source = inspect.getsource(func)
        fut = self.executor.submit(_subproc_wrapper, func_source, *args, **kwargs)
        return fut

    def submit_with_data_staging(
        self, func_source: str, data_files: dict[str, Path], *args, **kwargs
    ) -> Any:
        """
        Execute a function with large data files staged via chunked upload.

        Args:
            func_source: The function to execute
            data_files: Dictionary mapping remote filename to local file path
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            the task id of the function
        """
        # Stage all data files first
        staged_paths = {}
        for remote_name, local_path in data_files.items():
            staged_path = self._stage_file(local_path, remote_name)
            staged_paths[remote_name] = staged_path

        # Submit main function with staged file paths
        return self.submit(func_source, staged_paths, *args, **kwargs)

    def _stage_file(self, local_path: Path, remote_name: str) -> str:
        """
        Stage a local file to the HPC by chunked upload.

        Args:
            local_path: Local file path
            remote_name: Desired filename on HPC

        Returns:
            Remote file path where the file was staged
        """
        # Read file data
        with open(local_path, "rb") as f:
            file_data = f.read()

        # Generate unique staging path
        staging_id = uuid.uuid4().hex[:8]
        remote_path = f"/tmp/garden_staging/{staging_id}_{remote_name}"

        # Split into chunks
        chunk_size = 4_000_000  # 4MB chunks for safety
        chunks = self._split_into_chunks(file_data, chunk_size)

        print(f"Staging {local_path} to {remote_path} in {len(chunks)} chunks...")

        # Submit chunks sequentially through subprocess wrapper
        for i, chunk in enumerate(chunks):
            print(f"  Uploading chunk {i+1}/{len(chunks)}")
            # Get the source code of _write_chunk function
            import inspect  # noqa:

            chunk_func_source = inspect.getsource(_write_chunk)

            # Submit through subprocess wrapper
            future = self.executor.submit(
                _subproc_wrapper, chunk_func_source, remote_path, i, chunk, len(chunks)
            )
            # Wait for each chunk to complete before sending next
            result = future.result()
            if "error" in result:
                raise RuntimeError(f"Failed to upload chunk {i}: {result['error']}")

        print(f"  File staging complete: {remote_path}")
        return remote_path

    def _split_into_chunks(self, data: bytes, chunk_size: int) -> list[bytes]:
        """Split data into chunks of specified size."""
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunks.append(data[i : i + chunk_size])
        return chunks

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

    def decode_result_data(self, raw_data: str):
        """
        Decode base64 pickled result data when ASE is available.

        Args:
            raw_data: Base64 encoded pickled data

        Returns:
            Unpickled result object
        """
        try:
            import ase  # noqa
            import ase.atoms  # noqa
            import ase.cell  # noqa

            return pickle.loads(base64.b64decode(raw_data))
        except ImportError:
            raise ImportError(
                "ASE is required to decode result data containing ASE objects"
            )


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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
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


def _write_chunk(
    remote_path: str, chunk_index: int, chunk_data: bytes, total_chunks: int
):
    """
    Write a chunk of data to the staging area on HPC.

    Args:
        remote_path: Final path where the complete file should be assembled
        chunk_index: Index of this chunk (0-based)
        chunk_data: Raw bytes for this chunk
        total_chunks: Total number of chunks expected
    """
    import os
    from pathlib import Path
    import time

    def assemble_chunks(remote_path: str, total_chunks: int):
        """Assemble all chunks into the final file."""
        try:
            # Wait for all chunks to be written (up to 60 seconds)
            timeout = 60
            start_time = time.time()

            while time.time() - start_time < timeout:
                chunk_files = [
                    f"{remote_path}.chunk_{i:04d}" for i in range(total_chunks)
                ]
                if all(os.path.exists(f) for f in chunk_files):
                    break
                time.sleep(0.5)
            else:
                missing_chunks = [
                    i
                    for i in range(total_chunks)
                    if not os.path.exists(f"{remote_path}.chunk_{i:04d}")
                ]
                return {
                    "error": f"Timeout waiting for chunks. Missing: {missing_chunks}"
                }

            # Assemble chunks into final file
            with open(remote_path, "wb") as outfile:
                for i in range(total_chunks):
                    chunk_file = f"{remote_path}.chunk_{i:04d}"
                    with open(chunk_file, "rb") as infile:
                        outfile.write(infile.read())
                    # Clean up chunk file
                    os.remove(chunk_file)

            file_size = os.path.getsize(remote_path)
            return {
                "success": f"File assembled successfully at {remote_path} ({file_size} bytes)"
            }

        except Exception as e:
            return {"error": f"Failed to assemble chunks: {str(e)}"}

    # Create staging directory if it doesn't exist
    staging_dir = Path(remote_path).parent
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Write this chunk to a temporary file
    chunk_file = f"{remote_path}.chunk_{chunk_index:04d}"

    try:
        with open(chunk_file, "wb") as f:
            f.write(chunk_data)

        # If this is the last chunk, assemble the complete file
        if chunk_index == total_chunks - 1:
            return assemble_chunks(remote_path, total_chunks)
        else:
            return {"success": f"Chunk {chunk_index} written successfully"}

    except Exception as e:
        return {"error": f"Failed to write chunk {chunk_index}: {str(e)}"}


def _assemble_chunks(remote_path: str, total_chunks: int):
    """
    Assemble all chunks into the final file.

    Args:
        remote_path: Final path where the complete file should be assembled
        total_chunks: Total number of chunks expected
    """
    import os
    import time

    try:
        # Wait for all chunks to be written (up to 60 seconds)
        timeout = 60
        start_time = time.time()

        while time.time() - start_time < timeout:
            chunk_files = [f"{remote_path}.chunk_{i:04d}" for i in range(total_chunks)]
            if all(os.path.exists(f) for f in chunk_files):
                break
            time.sleep(0.5)
        else:
            missing_chunks = [
                i
                for i in range(total_chunks)
                if not os.path.exists(f"{remote_path}.chunk_{i:04d}")
            ]
            return {"error": f"Timeout waiting for chunks. Missing: {missing_chunks}"}

        # Assemble chunks into final file
        with open(remote_path, "wb") as outfile:
            for i in range(total_chunks):
                chunk_file = f"{remote_path}.chunk_{i:04d}"
                with open(chunk_file, "rb") as infile:
                    outfile.write(infile.read())
                # Clean up chunk file
                os.remove(chunk_file)

        file_size = os.path.getsize(remote_path)
        return {
            "success": f"File assembled successfully at {remote_path} ({file_size} bytes)"
        }

    except Exception as e:
        return {"error": f"Failed to assemble chunks: {str(e)}"}


def _cleanup_staged_files(*file_paths):
    """
    Clean up staged files after job completion.

    Args:
        *file_paths: Paths to files that should be cleaned up
    """
    import os

    cleaned = []
    errors = []

    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                cleaned.append(file_path)
        except Exception as e:
            errors.append(f"Failed to remove {file_path}: {str(e)}")

    return {"cleaned": cleaned, "errors": errors, "success": len(errors) == 0}
