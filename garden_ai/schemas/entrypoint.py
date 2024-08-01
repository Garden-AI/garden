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
    repo_name: str
    url: Url
    contributors: UniqueList[str] = Field(default_factory=list)


class PaperMetadata(BaseModel):
    title: str
    authors: UniqueList[str] = Field(default_factory=list)
    doi: str | None = None
    citation: str | None = None


class DatasetMetadata(BaseModel):
    title: str
    repository: str
    doi: str | None = None
    url: Url | None = None
    data_type: str | None = None

    # copied from old DatasetConnection
    @field_validator("repository")
    @classmethod
    def check_foundry(cls, v, values):
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
    model_identifier: str
    model_repository: str
    model_version: str | None = None


class EntrypointMetadata(BaseModel):
    """User-provided metadata about an entrypoint prior to its registration. \
    Passed to the `garden_entrypoint` decorator marking the entrypoint function.

    Attributes:
        title: A short title that describes the entrypoint.
        description: A longer free text description of this entrypoint.
        authors: A list of the authors of this entrypoint. You need at least one.
        short_name: This will be the name of the Python method that people call to invoke your entrypoint.
        year: When did you make this entrypoint? (Defaults to current year)
        tags: Tags to associate with the entrypoint for discoverability.
        repositories: List of related code repositories (``RepositoryMetadata``), like GitHub or GitLab repos.
        papers: List of related papers, like a paper (``PaperMetadata``) that describes the model you are publishing here.
        datasets: List of related datasets (``DatasetMetadata``) that is related to the entrypoint you are publishing.
    """

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


class RegisteredEntrypointMetadata(EntrypointMetadata):
    """Class containing complete (user- and garden-provided) metadata about an entrypoint.

    This corresponds to the EntrypointMetadataResponse schema on the backend.
    """

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

    def datacite_json(self) -> JsonStr:
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
