#!/usr/bin/env python3
import json
import pathlib
from unittest.mock import MagicMock, Mock, mock_open, patch

import docker  # type: ignore
import nbconvert
import pytest

import garden_ai.containers

JUPYTER_TOKEN = "bunchanumbers"


@pytest.fixture
def mock_docker_client():
    client = Mock(spec=docker.DockerClient)
    client.images.pull.return_value = None
    client.containers.run.return_value = Mock(spec=docker.models.containers.Container)
    return client


def test_start_container_with_notebook(mock_docker_client):
    path = pathlib.Path("/path/to/notebook.ipynb")
    base_image = "gardenai/fake-image:soonest"

    with patch("garden_ai.containers.JUPYTER_TOKEN", JUPYTER_TOKEN):
        container = garden_ai.containers.start_container_with_notebook(
            mock_docker_client, path, base_image
        )

    mock_docker_client.images.pull.assert_called_once_with(
        base_image, platform="linux/x86_64"
    )

    expected_volumes = {str(path): {"bind": f"/garden/{path.name}", "mode": "rw"}}

    expected_entrypoint = [
        "jupyter",
        "notebook",
        "--notebook-dir=/garden",
        f"--ServerApp.token={JUPYTER_TOKEN}",
        "--ip",
        "0.0.0.0",
        "--no-browser",
        "--allow-root",
    ]

    mock_docker_client.containers.run.assert_called_once_with(
        base_image,
        platform="linux/x86_64",
        detach=True,
        ports={"8888/tcp": 8888},
        volumes=expected_volumes,
        entrypoint=expected_entrypoint,
        stdin_open=True,
        tty=True,
        remove=True,
    )

    assert isinstance(container, Mock)


def test_build_notebook_session_image(mock_docker_client, mocker):
    # Mocking required values and paths
    notebook_path = pathlib.Path("/path/to/notebook.ipynb")
    base_image = "gardenai/fake-image:soonest"
    image_repo = "gardenai/fake-repo"
    notebook_content = '{"cells": []}'
    script_content = "print('Hello')"

    # Mocking nbconvert.PythonExporter()
    mock_exporter = MagicMock(spec=nbconvert.PythonExporter)
    mock_exporter.from_filename.return_value = (script_content, None)
    mock_python_exporter = patch("nbconvert.PythonExporter", return_value=mock_exporter)

    # Mocking open() to return our mocked notebook and script content
    m = mock_open(read_data=notebook_content)
    m.return_value.__iter__.return_value = notebook_content.splitlines()
    mock_open_func = patch("builtins.open", m)

    # Mocking inspect.getsource
    mock_getsource = patch("inspect.getsource", return_value="mock source code")

    # Mocking TemporaryDirectory and path related functions
    mock_temp_dir = MagicMock(spec=pathlib.Path)
    mock_temp_dir.__truediv__.return_value = notebook_path
    mock_temp_dir_context = patch(
        "garden_ai.containers.TemporaryDirectory", return_value=mock_temp_dir
    )

    with mock_python_exporter, mock_open_func, mock_getsource, mock_temp_dir_context:
        # Invoke function
        result = garden_ai.containers.build_notebook_session_image(
            client=mock_docker_client,
            notebook_path=notebook_path,
            base_image=base_image,
            image_repo=image_repo,
        )

    # Assertions
    mock_docker_client.images.pull.assert_called_once_with(
        base_image, platform="linux/x86_64"
    )
    mock_docker_client.images.build.assert_called()

    assert isinstance(result, Mock)


def test_extract_metadata_from_image(mock_docker_client):
    # Mock metadata returned from image
    mock_metadata = {
        "function_name_1": {"some": "metadata"},
        "function_name_2": {"other": "metadata"},
    }
    mock_metadata_str = json.dumps(mock_metadata)
    mock_container = MagicMock(spec=docker.models.containers.Container)
    mock_container.decode.return_value = mock_metadata_str
    mock_docker_client.containers.run.return_value = mock_container

    # Mock image object
    mock_image = MagicMock(spec=docker.models.images.Image)
    mock_image.id = "some_image_id"

    # Invoke function
    result = garden_ai.containers.extract_metadata_from_image(
        client=mock_docker_client, image=mock_image
    )

    # Assertions
    mock_docker_client.containers.run.assert_called_once_with(
        image=mock_image.id,
        entrypoint="/bin/sh",
        command=["-c", "cat /garden/metadata.json"],
        remove=True,
        detach=False,
    )
    assert result == mock_metadata


def test_push_image_to_public_repo(mock_docker_client):
    # Mock values
    repo_name = "username/myrepo"
    tag = "mytag"
    expected_image_location = f"docker.io/{repo_name}:{tag}"

    # Mock push logs from Docker client
    mock_push_logs = [
        {"status": "Preparing", "progress": "[==========>                    ]"},
        {"status": "Pushing", "progress": "[=============================>   ]"},
        {"status": "Pushed"},
    ]
    mock_docker_client.images.push.return_value = iter(mock_push_logs)

    # Mock image
    mock_image = MagicMock(spec=docker.models.images.Image)

    # Call function
    with patch("builtins.print") as mock_print:
        result = garden_ai.containers.push_image_to_public_repo(
            client=mock_docker_client, image=mock_image, repo_name=repo_name, tag=tag
        )

    # Assertions
    mock_image.tag.assert_called_once_with(repo_name, tag=tag)
    mock_docker_client.images.push.assert_called_once_with(
        repo_name, tag=tag, stream=True, decode=True
    )

    # Check if the logs were printed correctly
    mock_print.assert_any_call("Preparing - [==========>                    ]")
    mock_print.assert_any_call("Pushing - [=============================>   ]")
    mock_print.assert_any_call("Pushed")

    # Check the return value
    assert result == expected_image_location
