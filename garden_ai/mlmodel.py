from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field, field_validator, ConfigDict


class ModelRepository(Enum):
    HUGGING_FACE = "Hugging Face"
    GITHUB = "GitHub"


class DatasetConnection(BaseModel):
    """
    The ``DatasetConnection`` class represents metadata of external datasets \
    which publishers want to highlight as related to their Entrypoint. This \
    metadata (if provided) will be displayed with the Entrypoint as "related \
    work".

    These can be linked to an Entrypoint directly in the `EntrypointMetadata` or \
    via the ``@garden_entrypoint`` decorator with the `datasets` kwarg.

    Example:

        ```python
        my_relevant_dataset = DatasetConnection(
            title="Benchmark Dataset for Locating Atoms in STEM images",
            doi="10.18126/e73h-3w6n",
            url="https://foundry-ml.org/#/datasets/10.18126%2Fe73h-3w6n",
            repository="foundry",
        )
        my_metadata = EntrypointMetadata(
            title="...",
            # etc
        )

        @garden_entrypoint(metadata=my_metadata, datasets=[my_relevant_dataset])
        def my_entrypoint(*args, **kwargs):
            ...

        ```


    Attributes:
        title (str):
            A short and descriptive name of the dataset.
        doi (str):
            A digital identifier to the dataset.
        url (str):
            Location where the dataset can be accessed. If using foundry \
            dataset, both url and DOI must be provided.
        repository (str):
            The public repository where the dataset is hosted (e.g. "foundry", "github")
        data_type (str):
            Optional, the type of file of dataset.

    """

    title: str = Field(...)
    doi: Optional[str] = Field(None)
    url: str = Field(...)
    data_type: Optional[str] = Field(None)
    repository: str = Field(...)

    @field_validator("repository")
    @classmethod
    def check_foundry(cls, v, values, **kwargs):
        v = v.lower()  # case-insensitive
        if "url" in values.data and "doi" in values.data:
            if v == "foundry" and (
                values.data["url"] is None or values.data["doi"] is None
            ):
                raise ValueError(
                    "For a Foundry repository, both url and doi must be provided"
                )
        return v


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

    model_config = ConfigDict(protected_namespaces=())

    model_identifier: str = Field(...)
    model_repository: str = Field(...)
    model_version: Optional[str] = Field(None)
    datasets: List[DatasetConnection] = Field(default_factory=list)

    @field_validator("model_repository")
    @classmethod
    def must_be_a_supported_repository(cls, model_repository):
        if model_repository not in [mr.value for mr in ModelRepository]:
            raise ValueError("is not a supported flavor")
        return model_repository


def match_repo_type(url: str) -> ModelRepository:
    """Match url to the appropriate ModelRepository"""
    if "github.com" in url:
        return ModelRepository("GitHub")
    elif "huggingface.co" in url:
        return ModelRepository("Hugging Face")
    else:
        raise ValueError("Repository type is not supported.")
