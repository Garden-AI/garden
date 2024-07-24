from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

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


# protected_namespaces=() to allow model_* attribute names
class ModelMetadata(BaseModel, protected_namespaces=()):
    model_identifier: str
    model_repository: str
    model_version: str | None = None


class EntrypointMetadata(BaseModel):
    """Class containing user-provided metadata about an entrypoint."""

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


class RegisteredEntrypointMetadata(EntrypointMetadata):
    """Class containing user- and garden-provided metadata about an entrypoint"""

    doi: str
    doi_is_draft: bool

    func_uuid: UUID
    container_uuid: UUID
    base_image_uri: str
    full_image_uri: str
    notebook_url: Url
    function_text: str

    requirements: list[str] = Field(default_factory=list)
    test_functions: list[str] = Field(default_factory=list)
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
