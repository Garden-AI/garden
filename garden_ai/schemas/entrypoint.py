from pydantic import BaseModel, Field, field_validator

from .schema_utils import UniqueList, Url


class RepositoryMetadata(BaseModel):
    """Metadata for a code repository to associate with an entrypoint.

    Attributes:
        repo_name: The name of the repository.
        url: The URL of the repository.
        contributors: (optional) A list of contributors to the repository.
    """

    repo_name: str
    url: Url
    contributors: UniqueList[str] = Field(default_factory=list)


class PaperMetadata(BaseModel):
    """Metadata for a research paper to associate with an entrypoint.

    Attributes:
        title: The title of the paper.
        authors: A list of authors of the paper.
        doi: The Digital Object Identifier (DOI) of the paper, if available.
        citation: A formatted citation for the paper, if available.
    """

    title: str
    authors: UniqueList[str] = Field(default_factory=list)
    doi: str | None = None
    citation: str | None = None


class DatasetMetadata(BaseModel):
    """Metadata for a dataset to associate with an entrypoint.

    Attributes:
        title: The title of the dataset.
        repository: The name of the repository hosting the dataset.
        doi: The Digital Object Identifier (DOI) of the dataset, if available.
        url: The URL where the dataset can be accessed, if available.
        data_type: The type or format of the data, if specified.

    Note:
        For Foundry repositories, both url and doi must be provided.
    """

    title: str
    repository: str
    doi: str | None = None
    url: Url | None = None
    data_type: str | None = None

    @field_validator("repository")
    @classmethod
    def _check_foundry(cls, v, values):
        """Ensures that Foundry repositories have both URL and DOI provided."""
        v = v.lower()  # case-insensitive
        if "url" in values.data and "doi" in values.data:
            if v.strip() == "foundry" and (
                values.data["url"] is None or values.data["doi"] is None
            ):
                raise ValueError(
                    "For a Foundry repository, both url and doi must be provided"
                )
        return v


# protected_namespaces=() to allow model_* attribute names
class ModelMetadata(BaseModel, protected_namespaces=()):
    """Metadata for a machine learning model associated with an entrypoint.

    **Not meant to be instantiated directly by users.**

    Unlike other forms of related metadata, this is typically created and linked automatically. Intended usage is to pass the result of [create_connector][garden_ai.create_connector] to the `@entrypoint` decorator.

    Attributes:
        model_identifier: A unique identifier for the model.
        model_repository: The repository where the model is stored.
        model_version: The version of the model, if applicable.
    """  # noqa: E501

    model_identifier: str
    model_repository: str
    model_version: str | None = None
