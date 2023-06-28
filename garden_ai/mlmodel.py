import json
import os
import pathlib
import pickle
import shutil
from enum import Enum
from functools import lru_cache
from typing import List

import requests
import zipfile
import mlflow  # type: ignore
from mlflow.pyfunc import load_model  # type: ignore
from pydantic import BaseModel, Field, validator

from garden_ai import GardenConstants

MODEL_STAGING_DIR = pathlib.Path(GardenConstants.GARDEN_DIR) / "mlflow"
MODEL_STAGING_DIR.mkdir(parents=True, exist_ok=True)


class ModelUploadException(Exception):
    """Exception raised when an attempt to upload a model to ML Flow fails"""

    pass


class PipelineLoadScaffoldedException(Exception):
    """Exception raised when a user attempts to load model with the name SCAFFOLDED_MODEL_NAME"""

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


class ModelMetadata(BaseModel):
    """
    The ``ModelMetadata`` class represents all the metadata we want to \
    publicly expose about an ML model that has been registered with Garden.

    Attributes:
        model_name (str): A short and descriptive name of the model
        flavor (str): The framework used for this model. One of "sklearn", "tensorflow", or "torch".
        connections (List[DatasetConnection]):
            A list of dataset records that the model was trained on.
        user_email (str): The email address of the user uploading the model.
        full_name (str): The user_email and model_name together like "foo@example.edu/my_model"
        mlflow_name (str): The user_email and model_name together like "foo@example.edu-my_model"

    """

    model_name: str = Field(...)
    user_email: str = Field(...)
    flavor: str = Field(...)
    connections: List[DatasetConnection] = Field(default_factory=list)
    full_name: str = ""
    mlflow_name: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The / separator is canonical because it is nice for S3
        # and conveys that your email is a namespace.
        self.full_name = f"{self.user_email}/{self.model_name}"
        # But for local MLFlow purposes, use a - separator instead
        # because MLFlow does not like slashes.
        self.mlflow_name = f"{self.user_email}-{self.model_name}"

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


class LocalModel(ModelMetadata):
    """
    The ``LocalModel`` class represents a pre-trained ML model that a user wants to register with Garden.
    It has everything a ModelMetadata has, plus a local_path where the user says the model can be loaded from.

    Extra attributes:
        local_path (str):
            Where the model is located on disk. Can be a file or a directory depending on the flavor.

    """

    local_path: str = Field(...)


def stage_model_for_upload(local_model: LocalModel) -> str:
    """
    Parameters
    ----------
    local_model: the model to be staged into an MLModel directory

    Returns full path of model directory
    -------
    """
    flavor, local_path = local_model.flavor, local_model.local_path
    try:
        if flavor == ModelFlavor.SKLEARN.value and pathlib.Path(local_path).is_file:
            with open(local_path, "rb") as f:
                loaded_model = pickle.load(f)
                log_model_variant = mlflow.sklearn.log_model
        elif flavor == ModelFlavor.TENSORFLOW.value:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            # ignore cpu guard info on tf import require before tf import
            from tensorflow import keras  # type: ignore

            loaded_model = keras.models.load_model(local_path)
            log_model_variant = (
                mlflow.tensorflow.log_model
            )  # TODO explore artifact path and sigs
        elif flavor == ModelFlavor.PYTORCH.value and pathlib.Path(local_path).is_file:
            import torch  # type: ignore

            loaded_model = torch.load(local_path)
            log_model_variant = mlflow.pytorch.log_model  # TODO explore signatures
        else:
            raise ModelUploadException(f"Unsupported model flavor {flavor}")

        # Create a folder structure for an experiment called "local" if it doesn't exist
        # in the user's .garden directory
        mlflow.set_tracking_uri("file://" + str(MODEL_STAGING_DIR))
        experiment_name = "local"
        mlflow.set_experiment(experiment_name)
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        # The only way to derive the full directory path MLFlow creates is with this context manager.
        with mlflow.start_run(None, experiment_id) as run:
            experiment_id = mlflow.active_run().info.experiment_id
            artifact_path = "model"
            log_model_variant(
                loaded_model,
                artifact_path,
                registered_model_name=local_model.mlflow_name,
            )
            model_dir = os.path.join(
                str(MODEL_STAGING_DIR),
                experiment_id,
                run.info.run_id,
                "artifacts",
                artifact_path,
            )
    except IOError as e:
        raise ModelUploadException("Could not open file " + local_path) from e
    except (pickle.PickleError, mlflow.MlflowException) as e:
        raise ModelUploadException("Could not parse model at " + local_path) from e

    return model_dir


def clear_mlflow_staging_directory():
    path = str(MODEL_STAGING_DIR)
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


class _Model:
    def __init__(
        self,
        model_full_name: str,
    ):
        self.model = None
        self.full_name = model_full_name

        # raises error if user tries to load model with the name SCAFFOLDED_MODEL_NAME
        if self.full_name == GardenConstants.SCAFFOLDED_MODEL_NAME:
            raise PipelineLoadScaffoldedException("Invalid model name.")

        # Not taking MODEL_STAGING_DIR from constants
        # so that this class has no external dependencies
        self.staging_dir = pathlib.Path(os.path.expanduser("~/.garden")) / "mlflow"
        return

    def download_and_stage(
        self, presigned_download_url: str, full_model_name: str
    ) -> str:
        try:
            download_dir = self.staging_dir / full_model_name
            download_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as pe:
            print(f"Could not create model staging directory: {pe}")
            raise

        zip_filepath = str(download_dir / "model.zip")

        try:
            response = requests.get(presigned_download_url, stream=True)
            response.raise_for_status()
        except requests.RequestException as re:
            print(
                f"Could not download model from presigned url. URL: {presigned_download_url}. Error: {re}"
            )
            raise

        try:
            with open(zip_filepath, "wb") as f:
                f.write(response.content)
        except IOError as ioe:
            print(f"Failed to write model to disk at location {zip_filepath}: {ioe}")
            raise

        extraction_dir = download_dir / "unzipped"
        unzipped_path = str(download_dir / extraction_dir)

        try:
            with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
                zip_ref.extractall(unzipped_path)
        except (FileNotFoundError, zipfile.BadZipFile) as fe:
            print(f"Failed to unzip model directory from Garden model repository. {fe}")
            raise

        return unzipped_path

    @staticmethod
    def get_download_url(full_model_name: str) -> str:
        model_url_json = os.environ.get("GARDEN_MODELS", None)
        if not model_url_json:
            raise Exception(
                "GARDEN_MODELS environment variable was not set. Cannot download model."
            )
        try:
            model_url_dict = json.loads(model_url_json)
            return model_url_dict[full_model_name]
        except (json.JSONDecodeError, KeyError):
            print(
                f"Could not find url for model {full_model_name} in GARDEN_MODELS env var contents {model_url_json}"
            )
            raise

    # Duplicated in this class so that _Model is self-contained
    def clear_mlflow_staging_directory(self):
        path = str(self.staging_dir)
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

    def _lazy_load_model(self):
        """download and deserialize the underlying model, if necessary."""
        if self.model is None:
            download_url = self.get_download_url(self.full_name)
            local_model_path = self.download_and_stage(download_url, self.full_name)
            self.model = load_model(local_model_path, suppress_warnings=True)
            try:
                self.clear_mlflow_staging_directory()
            except Exception as e:
                print(
                    f"Could not clean up model staging directory. Check permissions on {self.staging_dir}"
                )
                print(str(e))
                pass
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

    Tip:
        This is meant to be used as a "default argument" in a `@step`-decorated \
        function. This allows the step to collect model-specific dependencies, \
        including any user-specified dependencies when the model was registered.

    Example:
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
        The object returned by this function waits as long as possible - i.e. \
        until the model actually needs to make a prediction - to actually \
        deserialize the registered model. This is done so that \
        ``Model('me@uni.edu-myModel/2)`` in a step (like above) an argument \
        default is lighter-weight when the function itself is serialized for \
        remote execution of a pipeline.

        This function is also memoized, so the same object (which, being lazy, \
        may or may not have actually loaded the model yet) will be returned if \
        it is called multiple times with the same model_full_name.
    """
    return _Model(full_model_name)
