#!/usr/bin/env python3
import pathlib
from unittest.mock import Mock, patch

import docker  # type: ignore
import pytest

from garden_ai.containers import start_container_with_notebook

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
        container = start_container_with_notebook(mock_docker_client, path, base_image)

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
