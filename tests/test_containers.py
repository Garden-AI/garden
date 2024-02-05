#!/usr/bin/env python3
import json
import pathlib
from unittest.mock import MagicMock, Mock, patch
from garden_ai.constants import GardenConstants

import docker  # type: ignore
import pytest
import io
import garden_ai.containers

JUPYTER_TOKEN = "bunchanumbers"
IMAGE_ID = "sha256:2d6e000f4f63e1234567a1234567890123456789a1234567890b1234567890c1"


@pytest.fixture
def mock_docker_client():
    client = Mock(spec=docker.DockerClient)
    client.images.pull.return_value = None
    client.containers.run.return_value = Mock(spec=docker.models.containers.Container)

    # Mock the APIClient and set it as the `api` attribute on the DockerClient mock
    api_client = MagicMock()
    api_client.build.return_value = [
        {"stream": "Step 1/3 : FROM gardenai/base:python-3.10-jupyter"},
        {"stream": " ---> Using cache"},
        {"stream": " ---> 2d6e000f4f63"},
        {"aux": {"ID": IMAGE_ID}},
        {"stream": "Successfully built 2d6e000f4f63"},
    ]

    image_mock = MagicMock()
    image_mock.id = IMAGE_ID
    client.images.get.return_value = image_mock

    client.api = api_client
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
        f"--NotebookApp.token={JUPYTER_TOKEN}",
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


@pytest.fixture
def mock_notebook_path(tmp_path):
    path = tmp_path / "test_notebook.ipynb"
    path.write_text('{"cells": []}')
    return path


def test_build_notebook_session_image_success(
    mock_docker_client, mock_notebook_path, mocker
):
    base_image = "gardenai/fake-image:soonest"

    # fmt: off
    with patch("nbconvert.PythonExporter.from_filename", return_value=('print("Hello, world!")', None),), \
         patch("pathlib.Path.read_text", return_value="# Test notebook"), \
         patch("inspect.getsource", return_value="fake source code"):
        image = garden_ai.containers.build_notebook_session_image(
            client=mock_docker_client,
            notebook_path=mock_notebook_path,
            base_image=base_image,
        )
    # fmt: on

    # Test that the Docker image was pulled
    mock_docker_client.images.pull.assert_called_with(
        base_image, platform="linux/x86_64"
    )

    # Test that the build was called
    mock_docker_client.api.build.assert_called()

    # Test that the function returned something successfully
    assert image is not None


def test_extract_metadata_from_image(mock_docker_client):
    # Mock metadata returned from image
    mock_metadata = {
        "function_name_1": {"some": "metadata"},
        "function_name_2": {"other": "metadata"},
    }

    mock_container_stdout = json.dumps(mock_metadata).encode("utf-8")
    mock_docker_client.containers.run.return_value = mock_container_stdout

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


def test_push_image_to_public_repo(mock_docker_client, capsys):
    repo_name = GardenConstants.GARDEN_ECR_REPO
    tag = "latest"
    image_id = "some_image_id"

    # Mock image object
    mock_image = MagicMock(spec=docker.models.images.Image)
    mock_image.id = image_id

    # Mock tag method
    mock_image.tag = MagicMock(return_value=True)

    # Mock ECR auth
    mock_credentials = {"username": "username", "password": "password"}

    # Mock push logs
    mock_push_logs = [
        {"status": "The push refers to repository [docker.io/username/myrepo]"},
        {"status": "Preparing", "progress": "Preparing..."},
        {
            "status": "Pushing",
            "progress": "[===>                                               ]",
        },
        {
            "status": "Pushed",
            "progress": "[==================================================>]",
        },
        {"status": "latest: digest: sha256:somedigest size: 5278"},
    ]
    mock_docker_client.images.push.return_value = iter(mock_push_logs)

    # Call the function without printing logs to test the return value
    full_repo_tag = garden_ai.containers.push_image_to_public_repo(
        client=mock_docker_client,
        image=mock_image,
        tag=tag,
        auth_config=mock_credentials,
    )

    # Assertions
    mock_image.tag.assert_called_once_with(repo_name, tag=tag)
    mock_docker_client.images.push.assert_called_once_with(
        repo_name, auth_config=mock_credentials, tag=tag, stream=True, decode=True
    )
    assert full_repo_tag == f"docker.io/{repo_name}:{tag}"
    # Capture the output
    captured = capsys.readouterr()

    # Assertions to ensure that the logs are in the captured stdout
    for log in mock_push_logs:
        if "progress" in log:
            expected_log = f"{log['status']} - {log['progress']}"
        else:
            expected_log = log["status"]
        assert expected_log in captured.out


@pytest.mark.parametrize(
    "requirements_file", ["requirements.txt", "environment.yml", None]
)
def test_build_image_with_dependencies(mock_docker_client, mocker, requirements_file):
    base_image = "gardenai/base:python-3.10-jupyter"
    requirements_path = pathlib.Path(requirements_file) if requirements_file else None

    # Mock the file read for requirements.txt
    file_content = (
        "some-package==1.0.0"
        if requirements_file == "requirements.txt"
        else "dependencies:\n  - numpy"
    )
    mocker.patch("pathlib.Path.read_text", return_value=file_content)

    # Prepare to capture the Dockerfile content
    dockerfile_content_capture = io.StringIO()

    # Mock the 'open' method on the specific Path object
    mock_path_open = mocker.patch.object(pathlib.Path, "open")
    mock_path_open.return_value.__enter__.return_value = dockerfile_content_capture

    image_id = garden_ai.containers.build_image_with_dependencies(
        client=mock_docker_client,
        base_image=base_image,
        requirements_path=requirements_path,
        pull=True,
    )

    # Test that the base image was pulled
    mock_docker_client.images.pull.assert_called_with(
        base_image, platform="linux/x86_64"
    )
    contents = dockerfile_content_capture.getvalue()
    # Test that the Dockerfile was created correctly per requirements type
    assert f"FROM {base_image}" in contents
    assert "RUN pip install garden-ai" in contents
    if requirements_file == "requirements.txt":
        assert f"COPY {requirements_file} /garden/{requirements_file}" in contents
        assert "RUN pip install -r /garden/requirements.txt" in contents
        assert "conda" not in contents
    elif requirements_file == "environment.yml":
        assert f"COPY {requirements_file} /garden/{requirements_file}" in contents
        assert (
            "RUN conda env update --name base --file /garden/environment.yml "
            in contents
        )
        assert "pip install -r" not in contents
    else:  # None
        assert "COPY" not in contents
        assert "pip install -r" not in contents
        assert "conda" not in contents

    # Test that the build was called
    mock_docker_client.api.build.assert_called()

    # Test that the function returned the correct image ID
    # (this is set in the mock_docker_client fixture)
    expected_image_id = IMAGE_ID
    assert image_id == expected_image_id


def test_handle_docker_errors():
    def test_func():
        return "Hello, World!"

    wrapped_test_func = garden_ai.containers.handle_docker_errors(test_func)

    # Test when no exception is raised
    assert wrapped_test_func() == "Hello, World!"

    # Test variations on DockerExceptions that are thrown when Garden can't access Docker
    error_message_output_pairs = [
        (
            "Error while fetching server API version: ConnectionRefusedError",
            "Could not connect to your local Docker daemon. Double check that Docker is running.",
        ),
        (
            "Error while fetching server API version: PermissionError",
            "It looks like your current user does not have permissions to use Docker.",
        ),
    ]

    for error_message, expected_output in error_message_output_pairs:
        with pytest.raises(garden_ai.containers.DockerStartFailure) as excinfo:

            def test_func_error():
                raise docker.errors.DockerException(error_message)

            wrapped_test_func_error = garden_ai.containers.handle_docker_errors(
                test_func_error
            )
            wrapped_test_func_error()
        assert expected_output in str(excinfo.value.helpful_explanation)

    # Test the case where it's a startup error, but not one of the common cases
    with pytest.raises(garden_ai.containers.DockerStartFailure) as excinfo:

        def test_func_error():
            raise docker.errors.DockerException(
                "Error while fetching server API version: SomeOtherError"
            )

        wrapped_test_func_error = garden_ai.containers.handle_docker_errors(
            test_func_error
        )
        wrapped_test_func_error()
    assert "SomeOtherError" in str(excinfo.value.original_exception)

    # Test a Docker error unrelated to startup. The docker.errors.DockerException should be raised as-is.
    with pytest.raises(docker.errors.DockerException):

        def test_func_error():
            raise docker.errors.DockerException("Some Docker error")

        wrapped_test_func_error = garden_ai.containers.handle_docker_errors(
            test_func_error
        )
        wrapped_test_func_error()
