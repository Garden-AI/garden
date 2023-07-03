from pathlib import Path

from garden_ai import Pipeline


class PipelineLoadException(Exception):
    """Exception raised when a Pipeline couldn't be loaded from the user's module"""

    pass


def load_pipeline_from_python_file(python_file: Path) -> Pipeline:
    """
    Dynamically import a pipeline object from a user's pipeline file.

    Parameters
    ----------
    python_file: Local path of user's pipeline code.

    Returns
    -------
    The Pipeline extracted from the file.
    """
    from garden_ai.utils._meta import _load_pipeline_from_python_file

    try:
        return _load_pipeline_from_python_file(python_file)
    except ValueError as e:
        raise PipelineLoadException(
            f"Could not find pipeline object in {python_file}."
        ) from e
    except Exception as e:
        raise PipelineLoadException(
            f"Failed to execute the Python code in {python_file}"
        ) from e

    raise PipelineLoadException(f"Could not find pipeline object in {python_file}")
