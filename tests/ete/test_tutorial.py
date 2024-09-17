"""An end to end test of the garden user tutorial."""

from pathlib import Path
from typing import Generator

import pexpect as px  # type: ignore
import pytest
from rich import print

from .utils import (
    parse_ntbk_name_from_garden_output,
    GardenProcessError,
    insert_doi_into_tutorial_ntbk,
    spawn,
)


@pytest.fixture
def tutorial_notebook(
    docker_check, create_garden, tmp_path
) -> Generator[tuple[Path, str], None, None]:
    """Start a container running the tutorial notebook using the cli.

    Creates a new garden as recommended in the tutorial.
    Inserts the doi of the garden into the tutorial notebook so it can be published.

    Yield the path to the notebook and the garden doi it can be published to.
    """
    # Create a new garden
    new_garden_doi = create_garden("Tutorial Test Garden")

    # Start the tutorial notebook
    print("[blue]Starting tutorial notebook container...")
    try:
        nb_proc = spawn(
            "notebook start --base-image 3.10-sklearn --tutorial",
            tmp_path,
        )
        nb_proc.expect("Do you want to proceed?")
        nb_proc.sendline("y")

        # Wait for the container to build and start..
        nb_proc.expect("Notebook started! .+", timeout=300)
        notebook_name = parse_ntbk_name_from_garden_output(nb_proc.after)

        # Shutdown the container
        nb_proc.sendcontrol("c")
        nb_proc.wait()
        assert not nb_proc.isalive()
    except px.ExceptionPexpect as e:
        raise GardenProcessError(f"notebook start process failed: {str(e)}") from e

    if notebook_name is None:
        raise ValueError("Could not get notebook name from garden output.")

    # Get the generated notebook file
    notebook_path = Path(tmp_path) / notebook_name
    assert notebook_path.exists(), f"Notebook file not found: {notebook_path}"

    # Insert the doi into the notebook so it can be published
    insert_doi_into_tutorial_ntbk(new_garden_doi, notebook_path)

    yield notebook_path, new_garden_doi


@pytest.mark.ete
def test_tutorial(garden_client_authed, tutorial_notebook):
    """This test takes a new tutorial notebook with a valid garden doi and publishes the notebook.

    If publishing the notebook succeeds, we run the entrypoint with test input.
    This
    """
    notebook_path, garden_doi = tutorial_notebook

    # Publish the notebook
    print("[blue]Publishing tutorial notebook...")
    publish_proc = spawn(f"notebook publish {notebook_path}")
    try:
        publish_proc.expect("Built image:", timeout=30)
        publish_proc.expect("Pushing image to repository:", timeout=30)
        publish_proc.expect("Successfully pushed image to:", timeout=30)
        publish_proc.expect("Added entrypoint", timeout=30)
        publish_proc.close()
    except px.ExceptionPexpect as e:
        raise GardenProcessError(f"Publish process failed: {str(e)}") from e

    # run entrypoint remotely, the final step in the tutorial
    print("[blue]Calling published entrypoint...")
    iris_data = [[5.1, 3.5, 1.4, 0.2]]
    garden = garden_client_authed.get_garden(garden_doi)
    result = garden.classify_irises(iris_data)
    assert result == [
        "setosa"
    ], "Did not get correct result from tutorial endpoint execution"

    print("[green]End to End test of tutoiral successful.")
    print("[blue]Cleaning up...")
