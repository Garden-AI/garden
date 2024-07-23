from uuid import UUID

from pydantic import Field, BaseModel
from datetime import datetime

from .schema_utils import UniqueList, Url


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
    """Schema for user-provided metadata about an entrypoint."""

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
    """Full entrypoint metadata"""

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
