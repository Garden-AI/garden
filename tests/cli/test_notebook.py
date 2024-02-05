import pytest
from unittest.mock import patch, Mock
from typer import Exit
from docker.errors import DockerException  # type: ignore

from garden_ai.app.notebook import DockerClientSession
from garden_ai.containers import (
    DockerStartFailure,
    DockerBuildFailure,
    DockerPreBuildFailure,
)


def test_docker_client_session_start_failure():
    mock_docker_exception = Mock(spec=DockerException)
    docker_start_failure = DockerStartFailure(
        mock_docker_exception, "turn it off and back on again"
    )

    # Does it trigger a clean typer.Exit when we can't connect to Docker?
    with pytest.raises(Exit) as exc_info:
        with patch(
            "garden_ai.app.notebook.get_docker_client", side_effect=docker_start_failure
        ):
            with DockerClientSession():
                # Just needed to trigger __enter__
                pass
    assert exc_info.value.exit_code == 1, "Exit code is not 1."


@pytest.mark.parametrize(
    "build_error_class", [DockerBuildFailure, DockerPreBuildFailure]
)
def test_docker_client_session_build_failure(capsys, build_error_class):
    build_error = build_error_class(reason="bad notebook", build_log=["a", "b", "c"])

    # Does it trigger a clean typer.Exit when Docker build fails?
    with pytest.raises(Exit) as exc_info:
        with patch("garden_ai.app.notebook.get_docker_client"):
            with DockerClientSession():
                raise build_error
    assert exc_info.value.exit_code == 1, "Exit code is not 1."

    # And does it print the expected error message?
    captured = capsys.readouterr()
    assert "Fatal Docker build error:" in captured.out, "Error message not in output."
