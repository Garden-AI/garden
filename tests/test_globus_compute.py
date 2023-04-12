import pytest

from garden_ai.globus_compute.containers import build_container, ContainerBuildException
from garden_ai.globus_compute.remote_functions import (
    register_pipeline,
    PipelineRegistrationException,
)


class MockException(Exception):
    pass


def test_build_container_happy_path(compute_client, pipeline_toy_example):
    container_uuid = build_container(compute_client, pipeline_toy_example)
    assert container_uuid == "d1fc6d30-be1c-4ac4-a289-d87b27e84357"


def test_build_container_where_build_fails(
    compute_client, pipeline_toy_example, mocker
):
    compute_client.get_container_build_status = mocker.Mock(return_value="failed")
    with pytest.raises(ContainerBuildException):
        build_container(compute_client, pipeline_toy_example)


def test_build_container_with_container_request_error(
    compute_client, pipeline_toy_example, mocker
):
    mocker.patch(
        "garden_ai.globus_compute.containers.GlobusAPIError", new=MockException
    )
    compute_client.build_container.side_effect = MockException
    with pytest.raises(ContainerBuildException):
        build_container(compute_client, pipeline_toy_example)


def test_register_pipeline_happy_path(compute_client, pipeline_toy_example):
    func_uuid = register_pipeline(
        compute_client, pipeline_toy_example, "fake_container_id"
    )
    assert func_uuid == "f9072604-6e71-4a14-a336-f7fc4a677293"


def test_register_pipeline_with_request_error(
    compute_client, pipeline_toy_example, mocker
):
    mocker.patch(
        "garden_ai.globus_compute.remote_functions.GlobusAPIError", new=MockException
    )
    compute_client.register_function.side_effect = MockException
    with pytest.raises(PipelineRegistrationException):
        register_pipeline(compute_client, pipeline_toy_example, "fake_container_id")
