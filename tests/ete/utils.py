from pathlib import Path
import re
import sys

import nbformat
import pexpect as px  # type: ignore


class NoBackendError(Exception):
    """Raised when the there is an error communicating with the backend."""

    def __init__(self, msg):
        super().__init__(msg)


class GardenProcessError(Exception):
    """Raised when there is an error running a garden as a sub-process."""

    def __init__(self, msg):
        super().__init__(msg)


def spawn(garden_command: str, cwd: Path = Path(__file__).parent) -> px.spawn:
    """Spawn a garden CLI process with the given garden_command.

    Example:
    ```
    # runs 'garden-ai docker check'
    proc = spawn("docker check")
    ```

    The garden command runs using the same python that is running pytest.
    This allows the commands to have the same environment as the test suite.

    Return a pexpect.spawn object.
    """
    proc = px.spawn(
        # Run the garden cli using the same python executable running pytest
        f"{sys.executable} -m garden_ai " + garden_command,
        cwd=cwd,
        encoding="utf-8",
    )

    proc.logfile_read = sys.stdout
    proc.logfile_send = sys.stderr

    return proc


def parse_ntbk_name_from_garden_output(garden_output: str) -> str | None:
    """Return the notebook name from a 'garden notebook start' command output string"""
    notebook_name_match = re.search(
        # matches and returns garden-generated ipynb file names like 1813-Crimson-Shimmering-Cacti.ipynb
        r"notebooks/(.+\.ipynb) .*",  # garden puts the notebook in ./notebooks/ in the container
        garden_output,
    )
    return (
        notebook_name_match.group(1).strip()
        if notebook_name_match is not None
        else None
    )


def insert_doi_into_tutorial_ntbk(garden_doi: str, notebook_path: Path):
    """Insert the given doi into correct cell in the tutorial notebook.

    This is necessary for the notebook to be publishable.
    See: https://garden-ai.readthedocs.io/en/latest/user_guide/tutorial/#step-1-create-a-garden
    """
    ntbk = nbformat.read(notebook_path, nbformat.NO_CONVERT)
    nbformat.validate(ntbk)

    for i, cell in enumerate(ntbk.cells):
        if cell["cell_type"] == "code" and "my_garden_doi = ''" in cell["source"]:
            new_cell = nbformat.v4.new_code_cell(
                source=f"my_garden_doi = '{garden_doi}'\n",
            )
            # validation fails if we don't delete the cell id
            del new_cell["id"]
            ntbk.cells[i] = new_cell
            break

    nbformat.write(ntbk, notebook_path, nbformat.NO_CONVERT)


def parse_doi(output: str) -> str | None:
    """Parse the doi from a successful 'garden-ai garden create' command output"""
    match = re.search(r"DOI:\s*(\S+)", output)
    if match:
        return match.group(1)
    return None


def clean_output(output: str) -> str:
    """Clean up the escape codes from garden CLI output"""
    # This regex matches ANSI escape codes
    ansi_escape = re.compile(r"(?:\x1B[@-_][0-?]*[ -/]*[@-~])")
    # Substitute ANSI escape codes with an empty string
    return ansi_escape.sub("", output)
