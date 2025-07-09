import pickle
import base64
from typing import Any

from globus_compute_sdk import Executor
from globus_compute_sdk.serialize import ComputeSerializer, CombinedCode

EDITH_EP_ID = "a01b9350-e57d-4c8e-ad95-b4cb3c4cd1bb"


class EdithExecutor:
    def __init__(self, endpoint_id=EDITH_EP_ID):
        self.endpoint_id = endpoint_id
        self.endpoint_config = {
            "worker_init": """
                # need to load openmpi to avoid 'no non PBS mpiexec available' error
                module load openmpi
                # path where globus-compute-endpoint lives
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

    def submit(self, func_source: str, *args, **kwargs) -> Any:
        """
        Execute a function on a Globus Compute endpoint using subprocess with conda environment.

        Args:
            func_source: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function execution
        """
        cleaned_source = self._cleanup_modal_decorators(func_source)

        # submit the function the to gcmu exector
        fut = self.executor.submit(_subproc_wrapper, cleaned_source, *args, **kwargs)
        result = fut.result()

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

        return result

    def _cleanup_modal_decorators(self, func_source):
        # Clean the function source by removing modal decorators
        lines = func_source.split("\n")
        cleaned_lines = []
        for line in lines:
            # Skip lines that are modal decorators
            if line.strip().startswith("@app.") or line.strip().startswith("@modal."):
                continue
            cleaned_lines.append(line)
        cleaned_source = "\n".join(cleaned_lines)
        return cleaned_source

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
        # Run in conda environment
        cmd = ["conda", "run", "-n", "torch-sim-edith", "python", script_path]
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
