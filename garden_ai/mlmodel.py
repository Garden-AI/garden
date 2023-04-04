import pathlib
import pickle
from functools import lru_cache
from typing import List

import mlflow  # type: ignore
from mlflow.pyfunc import load_model  # type: ignore
import os
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# ignore cpu guard info on tf import require before tf import
from tensorflow import keras  # noqa: E402

from garden_ai.utils.misc import read_conda_deps


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


class _Model:
    def __init__(
        self,
        model_full_name: str,
    ):
        self.model = None
        self.model_uri = f"models:/{model_full_name}"
        # extract dependencies without loading model into memory
        # make it easier for steps to infer
        conda_yml = mlflow.pyfunc.get_model_dependencies(self.model_uri, format="conda")
        python_version, conda_dependencies, pip_dependencies = read_conda_deps(
            conda_yml
        )
        self.python_version = python_version
        self.conda_dependencies = conda_dependencies
        self.pip_dependencies = pip_dependencies
        return

    def _lazy_load_model(self):
        """deserialize the underlying model, if necessary."""
        if self.model is None:
            # don't clutter current directory, especially if running locally
            local_store = pathlib.Path("~/.garden/mlflow").expanduser()
            local_store.mkdir(parents=True, exist_ok=True)
            self.model = load_model(
                self.model_uri, suppress_warnings=True, dst_path=local_store
            )
        return

    def predict(self, data):
        """Generate model predictions.

        The underlying model will be downloaded if it hasn't already.

        input data is passed directly to the underlying model via its respective
        ``predict`` method.

        Parameters
        ----------
        data : pd.DataFrame | pd.Series | np.ndarray | List[Any] | Dict[str, Any]
            Input data fed to the model

        Returns
        --------
        Results of model prediction

        """
        self._lazy_load_model()
        return self.model.predict(data)


@lru_cache
def Model(full_model_name: str) -> _Model:
    """Load a registered model from Garden-AI's (MLflow) tracking server.

    Tip: This is meant to be used as a "default argument" in a
    ``@step``-decorated function. This allows the step to collect model-specific
    dependencies, including any user-specified dependencies when the model was
    registered.

    Example:
    --------
        ```python
        import garden_ai
        from garden_ai import step
        ....
        # OK for preceding step to return only a DataFrame
        @step
        def run_inference(
            my_data: pd.DataFrame,  # no default
            my_model = garden_ai.Model("me@uni.edu-myModel/2"),  # NOTE: used as default
        ) -> MyResultType:
        '''Run inference on DataFrame `my_data`, returned by previous step.'''

            result = my_model.predict(my_data)
            return result
        ```

    Notes:
    ------
    The object returned by this function waits as long as possible - i.e. until
    the model actually needs to make a prediction - to actually deserialize the
    registered model. This is done so that ``Model('me@uni.edu-myModel/2)`` in a
    step (like above) an argument default is lighter-weight when the function itself
    is serialized for remote execution of a pipeline.

    This function is also memoized, so the same object (which, being lazy, may
    or may not have actually loaded the model yet) will be returned if it is
    called multiple times with the same model_full_name.
    """
    return _Model(full_model_name)
