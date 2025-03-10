"""Fixtures and their helpers for creating end to end tests."""

from collections.abc import Callable
import os

import globus_sdk
import globus_compute_sdk
import pexpect as px  # type: ignore
import pytest
import requests
from rich import print

from garden_ai import GardenClient
from garden_ai.constants import GardenConstants
from .utils import spawn, parse_doi, NoBackendError, GardenProcessError, clean_output


@pytest.fixture(scope="module")
def setup_env():
    """Setup the environment for running end-to-and and integration tests."""
    print("[blue]Setting up environment...")
    os.environ["GARDEN_ENV"] = os.environ.get("GARDEN_ENV", "dev")
    os.environ["GARDEN_DISABLE_BROWSER"] = os.environ.get(
        "GAREDEN_DISABLE_BROWSER", "1"
    )
    yield


@pytest.fixture(scope="module")
def docker_check(setup_env):
    """Ensure garden cli can interact with docker.

    Raises Value error when garden can't communicate with docker
    """
    print("[blue]Checking for docker...")
    try:
        docker_check = spawn("docker check")
        docker_check.wait()
        assert docker_check.exitstatus == 0, "Failed to communicate with docker."
    except Exception as e:
        pytest.skip(f"Skipping due to docker error: {str(e)}")


@pytest.fixture(scope="module")
def dev_backend(setup_env):
    """Ensure the dev api is reachable.

    Raises NoBackendError when communication with backend fails."""
    print("[blue]Pinging the backend...")
    try:
        res = requests.get(GardenConstants.GARDEN_ENDPOINT)
        res.raise_for_status()
    except Exception as e:
        raise NoBackendError(
            f"Failed to commucate with backend: {GardenConstants.GARDEN_ENDPOINT}"
        ) from e


@pytest.fixture(scope="module")
def garden_client_authed(dev_backend, setup_env):
    """Return a garden client that has authed with globus so it can talk to the backend.

    Uses GARDEN_CLIENT_ID and GARDEN_CLIENT_SECRET env vars to auth with globus if present.
    Otherwise, construcs a garden client using local auth credentials.
    """
    print("[blue]Creating authed GardenClient...")
    if client_id := os.environ.get("GARDEN_CLIENT_ID"):
        if client_secret := os.environ.get("GARDEN_CLIENT_SECRET"):
            # Set these to auth with globus compute automatically
            os.environ["GLOBUS_COMPUTE_CLIENT_ID"] = client_id
            os.environ["GLOBUS_COMPUTE_CLIENT_SECRET"] = client_secret

            # make the GardenClient
            auth_client = globus_sdk.ConfidentialAppAuthClient(client_id, client_secret)
            return GardenClient(auth_client=auth_client)

    return GardenClient()


@pytest.fixture(scope="module")
def authed(garden_client_authed):
    """Make sure there are valid auth credentials available for new GardenClients."""
    # Write the auth tokens to disc so other tests auth automatically
    tokens = garden_client_authed.auth_client.oauth2_client_credentials_tokens(
        requested_scopes=[
            globus_sdk.AuthLoginClient.scopes.openid,
            globus_sdk.AuthLoginClient.scopes.email,
            globus_sdk.GroupsClient.scopes.view_my_groups_and_memberships,
            globus_compute_sdk.Client.FUNCX_SCOPE,
            GardenClient.scopes.action_all,
        ],
    )
    garden_client_authed.auth_key_store.store(tokens)


@pytest.fixture
def create_garden(dev_backend, authed, request) -> Callable[[str], str]:
    """Return a Callable that takes a title, creates a new garden, and returns the doi.

    Tracks all created gardens and deletes them after the tests run.
    """
    created_garden_dois = []

    def _create_garden(title: str) -> str:
        print("[blue]Creating new Garden...")
        try:
            garden_create = spawn(
                "garden create "
                f"--title='{title}' "
                "--author='The Garden Test Suite' "
                "--contributor='The Test Tsar' "
                "--year 2024 "
                "--description='A garden created during an end to end test.' "
                "--tag='Test'"
            )
            garden_create.expect("Garden .* created", timeout=30)
            garden_create.wait()
            assert (
                garden_create.exitstatus == 0
            ), f"Garden create exited with non-zero status: {garden_create.exitstatus}"
        except px.ExceptionPexpect as e:
            raise GardenProcessError(f"garden create process failed: {str(e)}") from e

        output = clean_output(garden_create.read())
        doi = parse_doi(output)
        if doi is None:
            raise ValueError("Could not find doi of created garden in output.")

        created_garden_dois.append(doi)
        return doi

    # Delete any gardens that were created during a test after the test finishes
    request.addfinalizer(lambda: _cleanup_gardens(created_garden_dois))

    return _create_garden


def _cleanup_gardens(dois: list[str]):
    print(f"[blue]Cleaning up {len(dois)} Gardens...")
    for doi in dois:
        garden_delete = spawn(f"garden delete {doi}")
        garden_delete.expect("Are you sure you want to proceed?", timeout=30)
        garden_delete.sendline("y")
        garden_delete.expect("Garden .* has been deleted", timeout=30)
        garden_delete.close()
