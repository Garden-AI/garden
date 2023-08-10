import os
import pathlib
import pickle
import joblib  # type: ignore
import shutil
from enum import Enum
from typing import Optional, List
import functools
import mlflow  # type: ignore
from pydantic import BaseModel, Field, validator

from garden_ai import GardenConstants
from garden_ai._model import _Model


MODEL_STAGING_DIR = pathlib.Path(GardenConstants.GARDEN_DIR) / "mlflow"
MODEL_STAGING_DIR.mkdir(parents=True, exist_ok=True)


class ModelUploadException(Exception):
    """Exception raised when an attempt to upload a model to ML Flow fails"""

    pass


class SerializationFormatException(Exception):
    """Exception raised when a serialization format is not supported by a given flavor"""

    pass


class ModelNotFoundException(Exception):
    """Exception raised when an attempt to access a model that does not exist"""

    pass


class PipelineLoadScaffoldedException(Exception):
    """Exception raised when a user attempts to load model with the name SCAFFOLDED_MODEL_NAME"""

    pass


class ModelFlavor(Enum):
    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


class SerializeType(
    Enum
):  # May be of value to incorporate this in the ModelFlavor enum.
    """
    Flavors can interact with multiple serialization types.
    Constraints are enforced within the model staging process.
    """

    PICKLE = "pickle"
    JOBLIB = "joblib"
    KERAS = "keras"  # keras/tf native save format
    TORCH = "torch"  # torch native save format


class DatasetConnection(BaseModel):
    """
    The ``DataSetConnection`` class represents all the metadata we want to \
    publically expose about the datasets that can be utilized with this model.

    Attributes:
        title (str):
            A short and descriptive name of the dataset.
        doi (str):
            A digital identifier to the dataset.
        url (str):
            Location where the dataset can be accessed. If using foundry \
            dataset, both url and DOI must be provided.
        data_type (str):
            Optional, the type of file of dataset.

    """

    title: str = Field(...)
    doi: Optional[str] = Field(None)
    url: str = Field(...)
    data_type: Optional[str] = Field(None)


class ModelMetadata(BaseModel):
    """
    The ``ModelMetadata`` class represents all the metadata we want to \
    publicly expose about an ML model that has been registered with Garden.

    Attributes:
        model_name (str): A short and descriptive name of the model
        flavor (str): The framework used for this model. One of "sklearn", "tensorflow", or "torch".
        serialize_type (str): The serialization/packaging format used for the model.
        dataset (DatasetConnection):
            A dataset record that the model was trained on.
        user_email (str): The email address of the user uploading the model.
        full_name (str): The user_email and model_name together like "foo@example.edu/my_model"
        mlflow_name (str): The user_email and model_name together like "foo@example.edu-my_model"

    """

    model_name: str = Field(...)
    user_email: str = Field(...)
    flavor: str = Field(...)
    serialize_type: Optional[str] = None
    dataset: Optional[DatasetConnection] = Field(None)
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

    @validator("serialize_type")
    def must_be_a_supported_serialize_type(cls, serialize_type):
        """
        Validates the serialization type when provided by the user, as it is optional.
        """
        if serialize_type and serialize_type not in [s.value for s in SerializeType]:
            raise ValueError("is not a supported model serialization format")
        return serialize_type


class LocalModel(ModelMetadata):
    """
    The ``LocalModel`` class represents a pre-trained ML model that a user wants to register with Garden.
    It has everything a ModelMetadata has, plus a local_path where the user says the model can be loaded from.

    Extra attributes:
        local_path (str):
            Where the model is located on disk. Can be a file or a directory depending on the flavor.
        extra_paths (List[str]):
            Where the extra Python files for the model is located on disk. Pytorch model specific.
    """

    local_path: str = Field(...)
    extra_paths: List[str] = Field(default_factory=list)


def stage_model_for_upload(local_model: LocalModel) -> str:
    """
    Parameters
    ----------
    local_model: the model to be staged into an MLModel directory

    Returns full path of model directory
    -------
    """
    flavor, local_path, serialization_type, extra_paths = (
        local_model.flavor,
        local_model.local_path,
        local_model.serialize_type,
        local_model.extra_paths,
    )
    try:
        if flavor == ModelFlavor.SKLEARN.value and pathlib.Path(local_path).is_file:
            if (
                serialization_type == SerializeType.PICKLE.value
                or serialization_type is None  # default to pickle for sklearn
            ):
                with open(local_path, "rb") as f:
                    loaded_model = pickle.load(f)
                    log_model_variant = mlflow.sklearn.log_model
            elif serialization_type == SerializeType.JOBLIB.value:
                with open(local_path, "rb") as f:
                    loaded_model = joblib.load(f)
                    log_model_variant = mlflow.sklearn.log_model
            else:
                raise SerializationFormatException(
                    f"Unsupported serialization format of type {serialization_type} for flavor {flavor}"
                )
            metadata = {"garden_load_strategy": "sklearn"}
        elif flavor == ModelFlavor.TENSORFLOW.value:
            if (
                serialization_type == SerializeType.KERAS.value
                or serialization_type is None
            ):
                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
                # ignore cpu guard info on tf import require before tf import
                from tensorflow import keras  # type: ignore

                loaded_model = keras.models.load_model(local_path)
                log_model_variant = (
                    mlflow.tensorflow.log_model
                )  # TODO explore artifact path and sigs
            else:
                raise SerializationFormatException(
                    f"Unsupported serialization format of type {serialization_type} for flavor {flavor}"
                )
            metadata = {"garden_load_strategy": "pyfunc"}
        elif flavor == ModelFlavor.PYTORCH.value and pathlib.Path(local_path).is_file:
            if (
                serialization_type == SerializeType.TORCH.value
                or serialization_type is None
            ):
                import torch  # type: ignore

                if torch.cuda.is_available():
                    loaded_model = torch.load(local_path)
                else:
                    loaded_model = torch.load(local_path, map_location="cpu")
                for file in extra_paths:
                    path = pathlib.Path(file)
                    if not path.exists() or not path.is_file() or path.suffix != ".py":
                        raise ModelUploadException(
                            f"{path} is not a valid Python file. Please provide a valid Python file (.py)."
                        )
                log_model_variant = mlflow.pytorch.log_model  # TODO explore signatures
            else:
                raise SerializationFormatException(
                    f"Unsupported serialization format of type {serialization_type} for flavor {flavor}"
                )
            metadata = {"garden_load_strategy": "pytorch"}
        else:
            raise ModelUploadException(f"Unsupported model flavor {flavor}")

        if extra_paths and flavor != ModelFlavor.PYTORCH.value:
            raise ModelUploadException(
                f"Sorry, extra files are only supported for pytorch models. The {flavor} flavor is not supported."
            )
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
                code_paths=extra_paths,
                metadata=metadata,
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
        if item == ".trash":
            continue
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def trackcalls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.has_been_called = True
        return func(*args, **kwargs)

    wrapper.has_been_called = False
    return wrapper


@trackcalls
def Model(full_model_name: str) -> _Model:
    from garden_ai.utils.filesystem import PipelineLoadException
    from .local_data import get_local_model_by_name

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
        download and deserialize the registered model. This is done so that \
        ``Model('me@uni.edu-myModel/2)`` in a step (like above) an argument \
        default is lighter-weight when the function itself is serialized for \
        remote execution of a pipeline.
    """
    if not full_model_name:
        raise PipelineLoadException(
            "The parameters of Model() are empty. Please enter your registered model name in your pipeline.py"
        )
    if not get_local_model_by_name(full_model_name):
        raise ModelNotFoundException(f"No model with {full_model_name} exists.")
    if full_model_name == GardenConstants.SCAFFOLDED_MODEL_NAME:
        error_message = (
            "Failed to load model. It looks like you are using the placeholder model name from a scaffolded pipeline. "
            f"Please replace {GardenConstants.SCAFFOLDED_MODEL_NAME} in your pipeline.py"
            " with the name of a registered Garden model."
            "\nFor more information on how to use Garden, please read our docs: "
            "https://garden-ai.readthedocs.io/en/latest/"
        )
        raise PipelineLoadScaffoldedException(error_message)
    try:
        from __main__ import _Model
    except ImportError:
        # re-import only seems necessary for pytest
        from garden_ai._model import _Model  # type: ignore
        from garden_ai.utils._meta import redef_in_main

        redef_in_main(_Model)
        from __main__ import _Model

    return _Model(full_model_name)
