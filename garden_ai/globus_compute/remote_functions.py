from garden_ai.pipelines import Pipeline
from funcx import FuncXClient  # type: ignore
from globus_sdk import GlobusAPIError


class PipelineRegistrationException(Exception):
    """Exception raised when a container build request fails"""

    pass


def register_pipeline(
    funcx_client: FuncXClient,
    pipeline: Pipeline,
    container_uuid: str,
) -> str:
    try:
        func_uuid = funcx_client.register_function(
            pipeline._composed_steps, container_uuid=container_uuid, public=True
        )
    except GlobusAPIError as e:
        raise PipelineRegistrationException(
            "Could not register pipeline on Globus Compute"
        ) from e
    return func_uuid
