"""HPC Function class for executing functions on HPC systems via Globus Compute."""

from globus_compute_sdk import Executor

from garden_ai.hpc.utils import subproc_wrapper, wait_for_task_id


class HpcFunction:
    """
    Represents an HPC function that can be submitted to remote compute endpoints.

    Attributes:
        metadata: HPC function metadata including available deployments and endpoints
        endpoints: List of available endpoint configurations with name and gcmu_id
    """

    def __init__(self, metadata):
        self.metadata = metadata
        # Build list of unique endpoints from available_deployments
        seen_endpoints = {}
        for dep_info in metadata.available_deployments:
            endpoint_id = dep_info.endpoint_gcmu_id
            if endpoint_id not in seen_endpoints:
                seen_endpoints[endpoint_id] = {
                    "name": dep_info.endpoint_name,
                    "id": endpoint_id,
                }
        self.endpoints = list(seen_endpoints.values())

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"HPC functions must be submitted asynchronously.\n"
            f"Try:\n\tjob_id = {self.metadata.function_name}.submit("
            f"endpoint_id='{self.endpoints[0]['id'] if self.endpoints else 'ENDPOINT_ID'}', "
            f"*args, **kwargs)"
        )

    def submit(
        self,
        *args,
        endpoint_id: str | None = None,
        worker_init: str | None = None,
        user_endpoint_config: dict | None = None,
        task_id_timeout: int = 60,
        **kwargs,
    ) -> str:
        """
        Submit the HPC function for asynchronous execution.

        Args:
            *args: Positional arguments for the function
            endpoint_id: Globus Compute endpoint UUID (required)
            worker_init: Custom worker initialization script
            user_endpoint_config: Custom endpoint configuration dict
            task_id_timeout: Seconds to wait for task_id from globus-compute
            **kwargs: Keyword arguments for the function

        Returns:
            Globus Compute task ID (string UUID)

        Example:
            >>> garden = client.get_garden("10.26311/some-doi")
            >>> job_id = garden.my_hpc_function.submit(
            ...     endpoint_id=garden.my_hpc_function.endpoints[0]['id'],
            ...     arg1="value1",
            ...     arg2=42
            ... )
            >>> status = garden.get_job_status(job_id)
            >>> results = garden.get_results(job_id)
        """
        if endpoint_id is None:
            available = [e["name"] for e in self.endpoints] if self.endpoints else []
            raise ValueError(
                f"Must provide endpoint_id. Available endpoints: {available}"
            )

        # Find the deployment info for this endpoint
        deployment_info = next(
            (
                d
                for d in self.metadata.available_deployments
                if d.endpoint_gcmu_id == endpoint_id
            ),
            None,
        )

        if deployment_info is None:
            raise ValueError(
                f"No deployment found for endpoint {endpoint_id}. "
                f"Available endpoints: {[e['name'] for e in self.endpoints]}"
            )

        # Build user_endpoint_config
        config = user_endpoint_config or {}
        if worker_init and "worker_init" not in config:
            config["worker_init"] = worker_init

        # Add conda_env_path to kwargs for subproc_wrapper
        kwargs["conda_env_path"] = deployment_info.conda_env_path

        # Submit via Globus Compute Executor with subproc_wrapper
        with Executor(
            endpoint_id=endpoint_id,
            user_endpoint_config=config if config else None,
        ) as executor:
            # Submit function source text via subproc_wrapper
            future = executor.submit(
                subproc_wrapper, self.metadata.function_text, *args, **kwargs
            )

        # Wait for task_id and return
        task_id = wait_for_task_id(future, task_id_timeout)
        return task_id
