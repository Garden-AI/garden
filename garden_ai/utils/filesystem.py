from pathlib import Path
import importlib.util
from garden_ai import Pipeline


class PipelineLoadException(Exception):
    """Exception raised when a container build request fails"""

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

    module_name = python_file.stem
    spec = importlib.util.spec_from_file_location(module_name, python_file)

    if spec is None or spec.loader is None:
        raise PipelineLoadException("Could not load the module from file")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise PipelineLoadException("Could not execute the Python code") from e

    for obj_name in dir(module):
        obj = getattr(module, obj_name)
        if isinstance(obj, Pipeline):
            if obj.short_name is None:
                obj.short_name = obj_name
            return obj

    raise PipelineLoadException(f"Could not find pipeline object in {python_file}")
