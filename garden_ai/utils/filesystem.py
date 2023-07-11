from pathlib import Path

from garden_ai import Pipeline

from garden_ai.mlmodel import PipelineLoadScaffoldedException
from garden_ai.constants import GardenConstants


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
    except PipelineLoadScaffoldedException as e:
        error_message = (
            "Failed to load model. It looks like you are using the placeholder model name from a scaffolded pipeline. "
            f"Please replace {GardenConstants.SCAFFOLDED_MODEL_NAME} in your pipeline.py"
            " with the name of a registered Garden model."
            "\nFor more information on how to use Garden, please read our docs: "
            "https://garden-ai.readthedocs.io/en/latest/"
        )
        raise PipelineLoadScaffoldedException(error_message) from e
    except ValueError as e:
        raise PipelineLoadException(
            f"Could not find pipeline object in {python_file}."
        ) from e
    except Exception as e:
        raise PipelineLoadException(
            f"Failed to execute the Python code in {python_file}, {str(e)}"
        ) from e

    raise PipelineLoadException(f"Could not find pipeline object in {python_file}")
