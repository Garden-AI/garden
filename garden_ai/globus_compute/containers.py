from time import sleep
from enum import Enum
from rich.console import Console

from funcx import FuncXClient, ContainerSpec  # type: ignore
from globus_sdk import GlobusAPIError
from garden_ai.pipelines import Pipeline

console = Console()


class ContainerBuildException(Exception):
    """Exception raised when a container build request fails"""

    pass


class BuildStatus(str, Enum):
    queued = "queued"
    building = "building"
    ready = "ready"
    failed = "failed"


def build_container(funcx_client: FuncXClient, pipeline: Pipeline) -> str:
    name = str(pipeline.uuid)
    cs = ContainerSpec(
        name=name,
        pip=pipeline.pip_dependencies,
        python_version=pipeline.python_version,
        conda=pipeline.conda_dependencies,
    )

    try:
        container_uuid = funcx_client.build_container(cs)
    except Exception as e:
        raise ContainerBuildException(
            "Could not submit build request to Container Service"
        ) from e

    poll_until_container_is_built(funcx_client, container_uuid)
    return container_uuid


def poll_until_container_is_built(funcx_client, container_uuid):
    """
    Given a uuid of a container, blocks until that container is built.
    Prints a cool status indicator to the console while polling.
    """
    timeout_at = 1800
    i = 0
    with console.status(
        "[bold green]Building container. This operation times out after 30 minutes."
    ) as status:
        while i < timeout_at:
            try:
                status = funcx_client.get_container_build_status(container_uuid)
            except GlobusAPIError as e:
                raise ContainerBuildException(
                    "Lost connection with Container Service during build"
                ) from e
            # Update the end user twice a minute
            if i % 30 == 0:
                console.log(f"Current status is {status}")
            if status in [BuildStatus.ready, BuildStatus.failed]:
                break
            sleep(5)
            i += 5
        else:
            raise ContainerBuildException(
                f"Container Build Timeout after {timeout_at} seconds"
            )

    if status == BuildStatus.failed:
        try:
            build_result = funcx_client.get_container(
                container_uuid, container_type="docker"
            )
        except GlobusAPIError as e:
            raise ContainerBuildException(
                "ContainerService build failed. Could not retrieve error reason."
            ) from e
        error_message = build_result["build_stderr"]
        raise ContainerBuildException(
            "ContainerService build failed. Build error message:\n" + error_message
        )
