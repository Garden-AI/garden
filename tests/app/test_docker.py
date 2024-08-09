import pytest

from garden_ai.containers import DockerStartFailure
from garden_ai.app.docker import ASCII_FLOWER


@pytest.mark.cli
def test_no_args_prints_usage(
    cli_runner,
    app,
):
    cli_args = [
        "docker",
    ]

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    assert "Usage:" in result.output


@pytest.mark.cli
def test_check_prints_error_if_docker_fails(
    cli_runner,
    app,
    mocker,
):
    cli_args = [
        "docker",
        "check",
    ]

    mocker.patch(
        "garden_ai.app.docker.get_docker_client",
        side_effect=DockerStartFailure(Exception("Intentional Error for Testing")),
    )

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    assert "Garden can't access Docker on your computer." in result.output


@pytest.mark.cli
def test_check_prints_success_message_when_docker_works(
    cli_runner,
    app,
    mocker,
):
    cli_args = ["docker", "check"]

    mocker.patch(
        "garden_ai.app.docker.get_docker_client", return_value=mocker.MagicMock()
    )

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    assert ASCII_FLOWER in result.output
    assert "Happy Gardening!" in result.output
