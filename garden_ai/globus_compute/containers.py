from time import sleep, time
from enum import Enum

from globus_compute_sdk import Client, ContainerSpec  # type: ignore
from globus_sdk import GlobusAPIError
from garden_ai.pipelines import Pipeline
from garden_ai.app.console import console


class ContainerBuildException(Exception):
    """Exception raised when a container build request fails"""

    pass


class BuildStatus(str, Enum):
    queued = "queued"
    building = "building"
    ready = "ready"
    failed = "failed"


def build_container(compute_client: Client, pipeline: Pipeline) -> str:
    name = str(pipeline.doi)
    cs = ContainerSpec(
        name=name,
        pip=pipeline.pip_dependencies,
        python_version=pipeline.python_version,
        conda=pipeline.conda_dependencies,
    )

    try:
        container_uuid = compute_client.build_container(cs)
    except GlobusAPIError as e:
        raise ContainerBuildException(
            "Could not submit build request to Container Service"
        ) from e

    _poll_until_container_is_built(compute_client, container_uuid)
    return container_uuid


def _poll_until_container_is_built(compute_client: Client, container_uuid: str):
    """
    Given a uuid of a container, block until that container is built.
    """
    # Time out in 1800 seconds, which is 30 minutes
    timeout = 1800
    # Update the user twice a minute
    log_interval = 30
    polling_interval = 5

    start_time = time()
    last_log_time = start_time

    while True:
        try:
            status = compute_client.get_container_build_status(container_uuid)
        except GlobusAPIError as e:
            raise ContainerBuildException(
                "Lost connection with Container Service during build"
            ) from e

        if status == BuildStatus.ready:
            return

        if status == BuildStatus.failed:
            _raise_build_failure_exception(compute_client, container_uuid)

        if time() - start_time > timeout:
            raise ContainerBuildException(
                f"Container Build Timeout after {timeout} seconds"
            )

        if time() - last_log_time > log_interval:
            console.log(f"Current status is {status}")
            last_log_time = time()

        sleep(polling_interval)


def _raise_build_failure_exception(compute_client: Client, container_uuid: str):
    try:
        build_result = compute_client.get_container(
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
