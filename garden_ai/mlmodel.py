from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, validator


class ModelNotFoundException(Exception):
    """Exception raised when an attempt to access a model that does not exist"""

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
