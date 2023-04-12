from garden_ai.pipelines import Pipeline
from globus_compute_sdk import Client  # type: ignore
from globus_sdk import GlobusAPIError


class PipelineRegistrationException(Exception):
    """Exception raised when a container build request fails"""

    pass


def register_pipeline(
    compute_client: Client,
    pipeline: Pipeline,
    container_uuid: str,
) -> str:
    try:
        func_uuid = compute_client.register_function(
            pipeline._composed_steps, container_uuid=container_uuid, public=True
        )
    except GlobusAPIError as e:
        raise PipelineRegistrationException(
            "Could not register pipeline on Globus Compute"
        ) from e
    return func_uuid
