#!/usr/bin/env python3
import os
from pathlib import Path
import re
import sys

from dotenv import load_dotenv
import globus_sdk
import nbformat
import pexpect as px  # type: noqa
import pytest
import requests

from garden_ai import GardenClient, Garden


ETE_ENV_PATH = Path(f"{__file__}").parent / ".env"
load_dotenv(ETE_ENV_PATH)


class NoBackendError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


@pytest.fixture
def garden_client():
    """Return a garden client that has authed with globus so it can talk to the backend."""
    if client_id := os.environ.get("GARDEN_CLIENT_ID"):
        if client_secret := os.environ.get("GARDEN_CLIENT_SECRET"):
            client = globus_sdk.ConfidentialAppAuthClient(client_id, client_secret)
            return GardenClient(auth_client=client)
    raise ValueError(
        "Unable to create garden client without GARDEN_CLIENT_ID and GARDEN_CLIENT_SECRET"
    )


@pytest.fixture(autouse=True)
def setup_env(tmp_path):
    """Setup the environment for running end-to-and and integration tests."""
    os.environ["GARDEN_DIR"] = str(tmp_path)
    os.environ["GARDEN_ENV"] = "dev"
    yield


@pytest.fixture(autouse=True)
def docker_check(setup_env):
    """Ensure garden cli can interact with docker.

    Raises Value error when garden can't communicate with docker
    """
    docker_check = spawn_garden_command("docker check")
    try:
        docker_check.expect("Happy Gardening!", timeout=5)
    except Exception as e:
        raise ValueError(
            "Unable to interact with docker. Cannot continue the test."
        ) from e


@pytest.fixture(autouse=True)
def dev_backend_up():
    """Ensure the dev api is reachable.

    Raises NoBackendError when communication with the dev api fails."""
    res = requests.get("https://api-dev.thegardens.ai/")
    try:
        res.raise_for_status()
    except Exception as e:
        raise NoBackendError("Unable to commucate with dev backend.") from e


@pytest.fixture
def new_garden(garden_client, mock_GardenMetadata) -> Garden:
    """Create and yield a new garden, delete it afterward"""
    new_garden = garden_client.create_garden(mock_GardenMetadata)
    yield new_garden
    garden_client.delete_garden(new_garden.doi)


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.slow
def test_tutorial_end_to_end(garden_client, new_garden):
    print("Starting end-to-end test of tutorial notebook")
    garden_dir = os.environ.get("GARDEN_DIR")

    # Start the tutorial notebook
    nb_proc = spawn_garden_command(
        "notebook start --base-image 3.10-sklearn --tutorial"
    )
    nb_proc.expect("Do you want to proceed?")
    nb_proc.sendline("y")

    # Wait for the container to build and start..
    nb_proc.expect("Container started! .+", timeout=60)
    # Extract the notebook name
    notebook_name = parse_ntbk_name_from_garden_ouput(nb_proc.after.decode("utf-8"))

    # Shutdown the container
    nb_proc.sendcontrol("c")
    nb_proc.expect("Notebook has stopped.")
    nb_proc.wait()
    assert not nb_proc.isalive()

    if notebook_name is None:
        raise ValueError("Could not extract notebook name from output.")

    # insert garden doi into notebook
    notebook_path = Path(garden_dir) / notebook_name
    insert_doi_into_tutorial_ntbk(new_garden.doi, notebook_path)

    # Publish the notebook
    publish_proc = spawn_garden_command(
        f"notebook publish {notebook_path} --base-image 3.10-sklearn --doi {new_garden.doi}"
    )
    publish_proc.expect("Successfully pushed image to: ", timeout=60)

    # run entrypoint remotely
    iris_data = [[5.1, 3.5, 1.4, 0.2]]
    garden = garden_client.get_garden(new_garden.doi)
    result = garden.classify_irises(iris_data)
    assert result == ["setosa"]


def spawn_garden_command(
    command_str: str,
    work_dir: Path = Path(os.environ.get("GARDEN_DIR", "/tmp/garden")),
) -> px.spawn:
    """Spawn a garden process using pexpect.spawn in the given work_dir

    Example:
        spawn_garden_command("docker check") # executes garden-ai docker check
    """
    return px.spawn(
        f"{sys.executable} -m garden_ai " + command_str,
        cwd=work_dir,
    )


def parse_ntbk_name_from_garden_ouput(garden_output: str) -> str | None:
    """Return the notebook name from a 'garden notebook start' command output string"""
    notebook_name_match = re.search(
        # matches and returns garden-generated ipynb file names like 1813-Crimson-Shimmering-Cacti.ipynb
        r"notebooks/(.+\.ipynb) .*",
        garden_output,
    )
    if not notebook_name_match:
        return None
    return notebook_name_match.group(1).strip()


def insert_doi_into_tutorial_ntbk(garden_doi: str, notebook_path: Path):
    """Insert the given doi into the correct cell in a tutorial notebook.

    See: https://garden-ai.readthedocs.io/en/latest/user_guide/tutorial/#step-1-create-a-garden
    """
    ntbk = nbformat.read(notebook_path, nbformat.NO_CONVERT)
    nbformat.validate(ntbk)
    for cell in ntbk.cells:
        if cell["cell_type"] == "code" and "my_garden_doi = ''" in cell["source"]:
            cell = nbformat.v4.new_code_cell(
                source=f"my_garden_doi = '{garden_doi}'",
            )
    nbformat.write(ntbk, notebook_path, nbformat.NO_CONVERT)
