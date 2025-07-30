import base64
import inspect
import pickle
from typing import Any

from globus_compute_sdk import Executor
from globus_compute_sdk.serialize import CombinedCode, ComputeSerializer

from garden_ai.hpc_gardens.utils import subproc_wrapper  # noqa:

EDITH_EP_ID = "a01b9350-e57d-4c8e-ad95-b4cb3c4cd1bb"

DEFAULT_CONFIG = {
    "worker_init": """true;
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

        # Determine conda environment based on model parameter (3rd positional arg)
        model = args[2] if len(args) >= 3 else ""
        conda_env = (
            "torch-sim-edith-mace"
            if str(model).startswith("mace")
            else "torch-sim-edith"
        )

        # Add conda_env to kwargs for subproc_wrapper
        kwargs["conda_env"] = conda_env

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
