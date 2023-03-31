import pathlib
import pickle
from functools import lru_cache
from typing import List

import mlflow  # type: ignore
from mlflow.pyfunc import load_model  # type: ignore
import os
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# ignore cpu gaurd info on tf import require before tf import
from tensorflow import keras  # noqa: E402


class ModelUploadException(Exception):
    """Exception raised when an attempt to upload a model to ML Flow fails"""

    pass


def upload_model(
    model_path: str,
    model_name: str,
    user_email: str,
    flavor: str,
    extra_pip_requirements: List[str] = None,
) -> str:
    """Upload a model to Garden-AI's MLflow model registry.

    Parameters
    ----------
    model_path : str
        The file path specific to the model type to be uploaded, e.g. ``/path/to/my/model.pkl``
        or ``/path/to/my/tf_model/`` or ``/path/to/my/torch_model.pt``.
    model_name : str
        The name under which to register the uploaded model.
    user_email : str
        The email address of the user uploading the model.
    flavor : str
        The library the model was made with, e.g. ``sklearn`` or ``tensorflow``.
    extra_pip_requirements : List[str], optional
        A list of additional pip requirements needed to load and/or run the model.
        Defaults to None.

    Returns
    -------
    str
        The full name and version number of the uploaded model in the MLflow
        registry, which can be used to fetch the model with a call to ``Model(...)``.

    Raises
    ------
    ModelUploadException
        If an error occurs during the upload process, such as failure to open or
        parse the model, or failure to retrieve the latest version of the model.
    """
    full_model_name = f"{user_email}-{model_name}"
    try:
        if flavor == "sklearn" and pathlib.Path(model_path).is_file:
            with open(model_path, "rb") as f:
                loaded_model = pickle.load(f)
                mlflow.sklearn.log_model(
                    loaded_model,
                    user_email,
                    registered_model_name=full_model_name,
                    extra_pip_requirements=extra_pip_requirements,
                )
        elif flavor == "tensorflow" and pathlib.Path(model_path).is_dir:
            loaded_model = keras.models.load_model(model_path)
            mlflow.tensorflow.log_model(  # TODO explore artifact path, sigs, and HDf5
                loaded_model,
                user_email,
                registered_model_name=full_model_name,
                extra_pip_requirements=extra_pip_requirements,
            )
        elif (
            flavor == "pytorch" and pathlib.Path(model_path).is_file
        ):  # TODO explore signatures
            loaded_model = torch.load(model_path)
            mlflow.pytorch.log_model(
                loaded_model,
                user_email,
                registered_model_name=full_model_name,
                extra_pip_requirements=extra_pip_requirements,
            )
    except IOError as e:
        raise ModelUploadException("Could not open file " + model_path) from e
    except pickle.PickleError as e:
        raise ModelUploadException("Could not parse model at " + model_path) from e
    except mlflow.MlflowException as e:
        raise ModelUploadException("Could not upload model.") from e

    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(full_model_name, stages=["None"])
    if len(versions) == 0:
        raise ModelUploadException("Could not retrieve model version.")
    version_number = versions[0].version
    return f"{full_model_name}/{version_number}"


@lru_cache
def Model(full_model_name: str) -> mlflow.pyfunc.PyFuncModel:
    """Load a registered model from Garden-AI's (MLflow) tracking server.

    Tip: This is meant to be used as a "default argument" in a
    ``@step``-decorated function. This allows the pipeline to download the model
    as soon as the steps are initialized, _before_ any calls to the pipeline.

    Example:
    --------
    ```python
    import garden_ai
    from garden_ai import step
    ....
    # OK for preceding step to return only a DataFrame
    @step
    def run_inference(
        my_data: pd.DataFrame,
        my_model = garden_ai.Model("me@uni.edu-myModel/2"),  # NOTE: only downloads once
                                                             # when python evaluates the
                                                             # default(s)
    ) -> MyResultType:
    '''Run inference on DataFrame `my_data`, returned by previous step.'''

        result = my_model.predict(my_data)
        return result
    ```

    Notes:
    ------
    This is currently just a wrapper around `mlflow.pyfunc.load_model`. In the future
    this might implement smarter caching behavior, but for now the preferred usage is
    to use this function as a default value for some keyword argument.
    """
    model_uri = f"models:/{full_model_name}"
    local_store = pathlib.Path("~/.garden/mlflow").expanduser()
    local_store.mkdir(parents=True, exist_ok=True)
    # don't clutter the user's current directory with mystery mlflow directories
    return load_model(model_uri=model_uri, suppress_warnings=True, dst_path=local_store)
