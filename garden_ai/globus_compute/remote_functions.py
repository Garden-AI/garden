from globus_compute_sdk import Client  # type: ignore
from globus_sdk import GlobusAPIError

from garden_ai.pipelines import Pipeline
from garden_ai.utils.misc import inject_env_kwarg


class PipelineRegistrationException(Exception):
    """Exception raised when a container build request fails"""

    pass


def register_pipeline(
    compute_client: Client,
    pipeline: Pipeline,
    container_uuid: str,
) -> str:
    try:
        to_register = inject_env_kwarg(pipeline._composed_steps)
        func_uuid = compute_client.register_function(
            to_register, container_uuid=container_uuid, public=True
        )
    except GlobusAPIError as e:
        raise PipelineRegistrationException(
            "Could not register pipeline on Globus Compute"
        ) from e
    return func_uuid
