import pathlib
import pickle
from enum import Enum
from functools import lru_cache
from typing import List

import mlflow  # type: ignore
from mlflow.pyfunc import load_model  # type: ignore
import os
from pydantic import BaseModel, Field, validator

from garden_ai.utils.misc import read_conda_deps


class ModelUploadException(Exception):
    """Exception raised when an attempt to upload a model to ML Flow fails"""

    pass


class ModelFlavor(Enum):
    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


class DatasetConnection(BaseModel):
    """
    A first step towards the Accelerate Connection Schema
    """

    type: str = "dataset"
    relationship: str = "origin"
    doi: str = Field(...)
    repository: str = "Foundry"
    url: str = Field(...)


class LocalModel(BaseModel):
    """
    The ``LocalModel`` class represents a trained ML model
    that a user wants to register with Garden.

    Args:
    model_name (str) A short and descriptive name of the model
    flavor (str): The framework used for this model. One of "sklearn", "tensorflow", or "torch".
    extra_pip_requirements (List[str]), optional
        A list of additional pip requirements needed to load and/or run the model.
        Defaults to None.
    local_path (str): Where the model is located on disk. Can be a file or a directory depending on the flavor.
    user_email (str): The email address of the user uploading the model.

    """

    model_name: str = Field(...)
    flavor: str = Field(...)
    extra_pip_requirements: List[str] = Field(default_factory=list)
    local_path: str = Field(...)
    user_email: str = Field(...)
    connections: List[DatasetConnection] = Field(default_factory=list)
    namespaced_model_name: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.namespaced_model_name = f"{self.user_email}-{self.model_name}"

    @validator("flavor")
    def must_be_a_supported_flavor(cls, flavor):
        if flavor not in [f.value for f in ModelFlavor]:
            raise ValueError("is not a supported flavor")
        return flavor

    @validator("model_name")
    def must_be_a_valid_model_name(cls, model_name):
        is_valid = all(c.isalnum() or c == "-" or c == "_" for c in model_name)
        if not is_valid:
            error_message = (
                "is not a valid model name. "
                "Model names can only contain alphanumeric characters, hyphens, and underscores."
            )
            raise ValueError(error_message)
        return model_name


class RegisteredModel(BaseModel):
    """
    The ``RegisteredModel`` class represents all the metadata
    we want to publicly expose about an ML model that has been registered with Garden.

    Args:
    model_name (str): A short and descriptive name of the model
    version (str): Version string like "1" or "2" for this model.
    flavor (str): The framework used for this model. One of "sklearn", "tensorflow", or "torch".
    connections (List[DatasetConnection]):
        A list of dataset records that the model was trained on.
    local_path (str): Where the model is located on disk. Can be a file or a directory depending on the flavor.
    user_email (str): The email address of the user uploading the model.

    """

    model_name: str = Field(...)
    version: str = Field(...)
    user_email: str = Field(...)
    flavor: str = Field(...)
    connections: List[DatasetConnection] = Field(default_factory=list)
    namespaced_model_name: str = ""
    model_uri: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.namespaced_model_name = f"{self.user_email}-{self.model_name}"
        self.model_uri = f"{self.namespaced_model_name}/{self.version}"


def upload_to_model_registry(local_model: LocalModel) -> RegisteredModel:
    """Upload a model to Garden-AI's MLflow model registry.

    Parameters
    ----------
    local_model : LocalModel
        The model to upload.

    Returns
    -------
    RegisteredModel
        Includes the full model_uri, which can be used to fetch the model with a call to ``Model(...)``.

    Raises
    ------
    ModelUploadException
        If an error occurs during the upload process, such as failure to open or
        parse the model, or failure to retrieve the latest version of the model.
    """
    _push_model_to_registry(local_model)
    return _assemble_metadata(local_model)


def _push_model_to_registry(local_model: LocalModel):
    flavor, local_path = local_model.flavor, local_model.local_path
    try:
        if flavor == ModelFlavor.SKLEARN.value and pathlib.Path(local_path).is_file:
            with open(local_path, "rb") as f:
                loaded_model = pickle.load(f)
                log_model_variant = mlflow.sklearn.log_model
        elif flavor == ModelFlavor.TENSORFLOW.value and pathlib.Path(local_path).is_dir:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            # ignore cpu guard info on tf import require before tf import
            from tensorflow import keras  # type: ignore

            loaded_model = keras.models.load_model(local_path)
            log_model_variant = (
                mlflow.tensorflow.log_model
            )  # TODO explore artifact path, sigs, and HDf5
        elif flavor == ModelFlavor.PYTORCH.value and pathlib.Path(local_path).is_file:
            import torch  # type: ignore

            loaded_model = torch.load(local_path)
            log_model_variant = mlflow.pytorch.log_model  # TODO explore signatures
        else:
            raise ModelUploadException(f"Unsupported model flavor {flavor}")
        log_model_variant(
            loaded_model,
            local_model.user_email,
            registered_model_name=local_model.namespaced_model_name,
            extra_pip_requirements=local_model.extra_pip_requirements,
        )
    except IOError as e:
        raise ModelUploadException("Could not open file " + local_path) from e
    except pickle.PickleError as e:
        raise ModelUploadException("Could not parse model at " + local_path) from e
    except mlflow.MlflowException as e:
        raise ModelUploadException("Could not upload model.") from e


def _assemble_metadata(local_model: LocalModel) -> RegisteredModel:
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(
        local_model.namespaced_model_name, stages=["None"]
    )
    if len(versions) == 0:
        raise ModelUploadException("Could not retrieve model version.")
    version_number = versions[0].version
    return RegisteredModel(
        **local_model.dict(),
        version=version_number,
    )


class _Model:
    def __init__(
        self,
        model_full_name: str,
    ):
        self.model = None
        self.model_full_name = model_full_name
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
        """download and deserialize the underlying model, if necessary."""
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
