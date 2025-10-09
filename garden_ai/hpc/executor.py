import base64
import inspect
import pickle
from typing import Any

from globus_compute_sdk import Executor
from globus_compute_sdk.serialize import CombinedCode, ComputeSerializer

from garden_ai.hpc.utils import subproc_wrapper  # noqa:

# EDITH endpoint ID - kept for backwards compatibility
EDITH_EP_ID = "a01b9350-e57d-4c8e-ad95-b4cb3c4cd1bb"

DEFAULT_CONFIG = {
    "worker_init": """true;
module load openmpi;
export PATH=/usr/mpi/gcc/openmpi-4.1.7rc1/bin:$PATH;
export PATH=$PATH:/usr/sbin;
""",
}


class HpcExecutor(Executor):
    """
    Executor for running functions on HPC systems via Globus Compute.

    This executor wraps functions to run in subprocess with specific conda environments,
    handling Python version mismatches between the client and remote endpoint.
    """

    def __init__(
        self,
        *args,
        endpoint_id=None,
        user_endpoint_config=None,
        **kwargs,
    ):
        # Default to EDITH endpoint if none provided (backwards compatibility)
        if endpoint_id is None:
            endpoint_id = EDITH_EP_ID

        # Default to standard config if none provided
        if user_endpoint_config is None:
            user_endpoint_config = DEFAULT_CONFIG

        # vary the user_endpoint_config to try to force separate PBS jobs
        if "user_endpoint_config" in kwargs and kwargs["user_endpoint_config"]:
            import time

            config = kwargs["user_endpoint_config"].copy()
            # Add a unique comment to make each job "different"
            if "worker_init" in config:
                config["worker_init"] += f"\n# Job timestamp: {time.time()}"
            kwargs["user_endpoint_config"] = config

        super().__init__(
            *args,
            endpoint_id=endpoint_id,
            user_endpoint_config=user_endpoint_config,
            **kwargs,
        )
        self.serializer = ComputeSerializer(strategy_code=CombinedCode())

    def submit(self, func, *args, **kwargs) -> Any:  # type: ignore[override]
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

        fut = super().submit(subproc_wrapper, func_source, *args, **kwargs)
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
