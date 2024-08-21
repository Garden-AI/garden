from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, PrivateAttr, field_validator

from .datacite import (
    Creator,
    DataciteSchema,
    Description,
    Identifier,
    Subject,
    Title,
    Types,
)
from .schema_utils import JsonStr, UniqueList, Url


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


class EntrypointMetadata(BaseModel):
    """User-provided metadata about an entrypoint prior to its registration.

    This class is used to define metadata for an entrypoint function, which is passed to the [@entrypoint decorator][garden_ai.entrypoint]. It encapsulates descriptive information about an entrypoint, including its name, authorship, and related work.

    Attributes:
        title: A short title that describes the entrypoint.
        authors: A list of the authors of this entrypoint. At least one author is required.
        short_name: The name of the Python method that users will call to invoke your entrypoint. If not provided, it defaults to the name of the decorated function. Must be a valid Python identifier.
        description: A longer free text description of this entrypoint.
        year: The year the entrypoint was created. Defaults to the current year.
        tags: Tags to associate with the entrypoint for discoverability.
        models: List of ModelMetadata objects describing models used by this entrypoint. Can also be specified in the @entrypoint decorator.
        repositories: List of RepositoryMetadata objects describing related code repositories, such as GitHub or GitLab repos. Can also be specified in the @entrypoint decorator.
        papers: List of PaperMetadata objects describing related papers, e.g., papers that describe the model you are publishing. Can also be specified in the @entrypoint decorator.
        datasets: List of DatasetMetadata objects describing related datasets used or produced by this entrypoint. Can also be specified in the @entrypoint decorator.

    Note:
        Only 'title' and 'authors' are required fields for citability. All other fields are optional but encouraged for better discoverability of your work.

    Example:
        ```python
        metadata = EntrypointMetadata(
            title="Iris Classifier",
            authors=["Alice Researcher"],
            description="A simple iris classification model",
            tags=["botany", "classification", "iris"]
        )
        @entrypoint(metadata=metadata)
        def classify_iris(data):
            # Function implementation
            return results
        ```
    """  # noqa: E501

    # only title and authors are hard requirements
    title: str
    authors: UniqueList[str]

    description: str | None = None
    year: str = Field(default_factory=lambda: str(datetime.now().year))
    short_name: str | None = None
    tags: UniqueList[str] = Field(default_factory=list)
    models: list[ModelMetadata] = Field(default_factory=list)
    repositories: list[RepositoryMetadata] = Field(default_factory=list)
    papers: list[PaperMetadata] = Field(default_factory=list)
    datasets: list[DatasetMetadata] = Field(default_factory=list)
    # private; populated directly by decorators
    _test_functions: list[str] = PrivateAttr(default_factory=list)
    _target_garden_doi: str | None = None
    _function_text: str | None = None

    @field_validator("short_name")
    @classmethod
    def must_be_valid_identifier(cls, short_name: str) -> str:
        assert (
            short_name.isidentifier()
        ), "short_name must be a valid python variable name."
        return short_name


class RegisteredEntrypointMetadata(EntrypointMetadata):
    """Represents the metadata needed to completely define a registered entrypoint.

    **Not meant to be instantiated directly by users.** See [EntrypointMetadata][garden_ai.EntrypointMetadata] for user-provided metadata.

    This class extends EntrypointMetadata with additional fields that are populated by the client during the entrypoint registration process.

    Attributes:
        doi: The Digital Object Identifier (DOI) assigned to this entrypoint. Provided by Garden.
        doi_is_draft: Whether the DOI is still in the default "draft" state.
        short_name: The callable name of the entrypoint function.
        test_functions: List of source code of test functions associated with this entrypoint. See also: [@entrypoint_test][garden_ai.entrypoint_test]
        requirements: List of pip package requirements for this entrypoint.
        func_uuid: UUID of the registered function in Globus Compute.
        container_uuid: UUID of the registered container image for this entrypoint.
        base_image_uri: URI of the base Docker image used for this entrypoint.
        full_image_uri: URI of the final Docker image for this entrypoint.
        notebook_url: URL to the notebook source of this entrypoint.
        function_text: Source code of the entrypoint function.
        owner_identity_id: Globus Auth UUID of the entrypoint's owner.
        id: Internal identifier for the entrypoint.

    """  # noqa: E501

    doi: str
    doi_is_draft: bool = True

    short_name: str
    test_functions: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)

    func_uuid: UUID
    container_uuid: UUID
    base_image_uri: str
    full_image_uri: str
    notebook_url: Url
    function_text: str

    owner_identity_id: UUID | None = None
    id: int | None = None

    def _datacite_json(self) -> JsonStr:
        """Convert metadata into a DataCite-schema-compliant JSON string."""
        return DataciteSchema(
            identifiers=[Identifier(identifier=self.doi, identifierType="DOI")],
            types=Types(resourceType="Pretrained AI/ML Model", resourceTypeGeneral="Software"),  # type: ignore
            creators=[Creator(name=name) for name in self.authors],
            titles=[Title(title=self.title)],
            publisher="thegardens.ai",
            publicationYear=self.year,
            subjects=[Subject(subject=tag) for tag in self.tags],
            descriptions=(
                [
                    Description(description=self.description, descriptionType="Other")  # type: ignore
                ]
                if self.description
                else None
            ),
        ).model_dump_json()
