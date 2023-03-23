from pathlib import Path
import importlib.util
import sys
from garden_ai import Pipeline


class PipelineLoadException(Exception):
    """Exception raised when a container build request fails"""

    pass


def load_pipeline_from_python_file(python_file: Path, pipeline_name: str) -> Pipeline:
    """
    Dynamically import a pipeline object from a user's pipeline file.

    Parameters
    ----------
    python_file: Local path of user's pipeline code.
    pipeline_name: The name of the pipeline object in the file.

    Returns
    -------
    The Pipeline extracted from the file.
    """

    module_name = python_file.stem
    spec = importlib.util.spec_from_file_location(module_name, python_file)

    if spec is None or spec.loader is None:
        raise PipelineLoadException("Could not load the module from file")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise PipelineLoadException("Could not execute the Python code") from e

    pipeline = getattr(module, pipeline_name, None)
    if not pipeline or not isinstance(pipeline, Pipeline):
        raise PipelineLoadException(
            f"Could not find pipeline named {pipeline_name} in {python_file}"
        )

    return pipeline
