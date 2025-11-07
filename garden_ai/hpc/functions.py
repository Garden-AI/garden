"""HPC Function class for executing functions on HPC systems via Globus Compute."""

import logging
from typing import TYPE_CHECKING, Any, TypeVar

from groundhog_hpc.future import GroundhogFuture

if TYPE_CHECKING:
    from garden_ai.client import GardenClient
else:
    GardenClient = TypeVar("GardenClient")

from concurrent.futures import Future

from garden_ai.hpc.groundhog import groundhog_in_harness, load_function_from_source
from garden_ai.hpc.utils import wait_for_task_id
from garden_ai.schemas.hpc import HpcFunctionMetadata, HpcInvocationCreateRequest

logger = logging.getLogger(__name__)


class HpcFunction:
    """
    Represents an HPC function that can be submitted to remote compute endpoints.

    Attributes:
        metadata: HPC function metadata including available deployments and endpoints
        endpoints: List of available endpoint configurations with name and gcmu_id
    """

    def __init__(
        self, metadata: HpcFunctionMetadata, client: GardenClient | None = None
    ):
        self.metadata = metadata
        self._client = client

        self._groundhog_function = load_function_from_source(
            self.metadata.function_text, self.metadata.function_name
        )
        self.endpoints = metadata.available_endpoints

    @property
    def client(self) -> GardenClient:
        return self._client or self._get_garden_client()

    def _get_garden_client(self) -> GardenClient:
        from garden_ai import GardenClient

        return GardenClient()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"HPC functions cannot be called directly. Use one of these methods instead:\n"
            f"1. For asynchronous execution (returns a Future with globus compute task_id):\n"
            f"\tjob = {self.metadata.function_name}.submit("
            f"*args, endpoint='{self.endpoints[0]['id'] if self.endpoints else 'ENDPOINT_ID'}', "
            "**kwargs)\n"
            f"\tresult = job.result()  # blocks until completion\n"
            f"2. For synchronous execution (blocks until completion):\n"
            f"\tresult = {self.metadata.function_name}.remote("
            f"*args, endpoint='{self.endpoints[0]['id'] if self.endpoints else 'ENDPOINT_ID'}', "
            "**kwargs)"
        )

    def _create_invocation_logger(self):
        """Create a callback that logs the invocation to the backend when the future completes."""

        def log_invocation(future: GroundhogFuture):
            try:
                endpoint_id = future.endpoint
                config = future.user_endpoint_config or {}
                task_id = future.task_id or wait_for_task_id(
                    future._original_future, timeout=120
                )
                invocation_request = HpcInvocationCreateRequest(
                    function_id=self.metadata.id,
                    endpoint_gcmu_id=endpoint_id,
                    globus_task_id=task_id,
                    user_endpoint_config=config,
                )
                self.client.backend_client.create_hpc_invocation(invocation_request)
            except Exception as e:
                # Don't fail the submission if logging fails
                logger.warning(f"Failed to log HPC invocation: {e}")

        return log_invocation

    def submit(
        self, *args, endpoint=None, walltime=None, user_endpoint_config=None, **kwargs
    ) -> Future:
        if endpoint is not None and endpoint not in self.endpoints:
            logger.warning(
                f"This function has not been deployed to {endpoint} before and may not work as expected."
            )

        with groundhog_in_harness():
            future = self._groundhog_function.submit(
                *args,
                endpoint=endpoint,
                walltime=walltime,
                user_endpoint_config=user_endpoint_config,
                **kwargs,
            )
            # Register callback to log invocation when the future completes
            future.add_done_callback(self._create_invocation_logger())
            return future

    def remote(
        self, *args, endpoint=None, walltime=None, user_endpoint_config=None, **kwargs
    ) -> Any:
        future: GroundhogFuture = self.submit(
            *args,
            endpoint=endpoint,
            walltime=walltime,
            user_endpoint_config=user_endpoint_config,
            **kwargs,
        )
        return future.result()

    def local(self, *args, **kwargs):
        return self._groundhog_function.local(*args, **kwargs)
