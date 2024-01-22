from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field, validator


class ModelRepository(Enum):
    HUGGING_FACE = "Hugging Face"
    GITHUB = "GitHub"


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
        repository (str):
            The public repository where the dataset is hosted
        data_type (str):
            Optional, the type of file of dataset.

    """

    title: str = Field(...)
    doi: Optional[str] = Field(None)
    url: str = Field(...)
    data_type: Optional[str] = Field(None)
    repository: str = Field(...)


class ModelMetadata(BaseModel):
    """
    The ``ModelMetadata`` class represents metadata about an ML model published  \
    on a public model repository used in an Entrypoint.

    Attributes:
        model_identifier (str): A short and descriptive name of the model
        model_repository (ModelRepository): The repository the model is published on.
        model_version (str): A version identifier
        datasets (DatasetConnection):
            One or more dataset records that the model was trained on.
    """

    model_identifier: str = Field(...)
    model_repository: str = Field(...)
    model_version: Optional[str] = Field(None)
    datasets: List[DatasetConnection] = Field(default_factory=list)

    @validator("model_repository")
    def must_be_a_supported_repository(cls, model_repository):
        if model_repository not in [mr.value for mr in ModelRepository]:
            raise ValueError("is not a supported flavor")
        return model_repository
