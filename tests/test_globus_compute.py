import pytest

from garden_ai.globus_compute.containers import build_container, ContainerBuildException


def test_build_container_happy_path(funcx_client, pipeline_toy_example):
    container_uuid = build_container(funcx_client, pipeline_toy_example)
    assert container_uuid == "d1fc6d30-be1c-4ac4-a289-d87b27e84357"


def test_build_container_where_build_fails(funcx_client, pipeline_toy_example, mocker):
    funcx_client.get_container_build_status = mocker.Mock(
        return_value="failed"
    )
    with pytest.raises(ContainerBuildException):
        build_container(funcx_client, pipeline_toy_example)


def test_build_container_with_container_request_error(funcx_client, pipeline_toy_example, mocker):
    mock_exception = mocker.MagicMock(Exception)
    mocker.patch("garden_ai.globus_compute.containers.GlobusAPIError", new=mock_exception)
    funcx_client.build_container.side_effect = mock_exception
    with pytest.raises(ContainerBuildException):
        build_container(funcx_client, pipeline_toy_example)
