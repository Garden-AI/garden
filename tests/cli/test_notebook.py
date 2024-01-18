import pytest
from unittest.mock import patch, Mock
from typer import Exit
from docker.errors import DockerException

from garden_ai.app.notebook import DockerClientSession, DockerStartFailure
from garden_ai.containers import DockerStartFailure


def test_docker_client_session():
    mock_docker_exception = Mock(spec=DockerException)
    docker_start_failure = DockerStartFailure(
        mock_docker_exception, "turn it off and back on again"
    )

    # Does it trigger a clean typer.Exit when we can't connect to Docker?
    with pytest.raises(Exit) as exc_info:
        with patch(
            "garden_ai.app.notebook.get_docker_client", side_effect=docker_start_failure
        ):
            with DockerClientSession() as docker_client:
                # Just needed to trigger __enter__
                pass
    assert exc_info.value.exit_code == 1, "Exit code is not 1."
