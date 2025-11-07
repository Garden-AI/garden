import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from groundhog_hpc.function import Function
from groundhog_hpc.utils import import_user_script


def load_function_from_source(contents: str, name) -> Function:
    """Load a groundhog function from script contents by writing to a temporary file.

    Args:
        contents: The script contents as a string
        name: The name of the function to load from the script

    Returns:
        The loaded groundhog Function instance
    """
    # Create a temporary file to write the script contents
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
    ) as temp_file:
        temp_file.write(contents)
        script_path = Path(temp_file.name)

    with groundhog_script_path(script_path):
        module = import_user_script(f"_{name}_module", script_path)
        obj = getattr(module, name)
    return obj


@contextmanager
def groundhog_script_path(script_path: Path):
    """temporarily set the GROUNDHOG_SCRIPT_PATH environment variable"""
    script_path = Path(script_path).resolve()
    try:
        # set this while exec'ing so the Function objects can template their shell functions
        os.environ["GROUNDHOG_SCRIPT_PATH"] = str(script_path)
        yield
    finally:
        del os.environ["GROUNDHOG_SCRIPT_PATH"]


@contextmanager
def groundhog_in_harness():
    """Simulate running in a @hog.harness function to enable remote execution"""
    try:
        os.environ["GROUNDHOG_IN_HARNESS"] = str(True)
        yield
    finally:
        del os.environ["GROUNDHOG_IN_HARNESS"]
