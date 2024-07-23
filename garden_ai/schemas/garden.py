from datetime import datetime
from uuid import UUID

from pydantic import Field, BaseModel

from .schema_utils import UniqueList


class GardenMetadata(BaseModel):
    doi: str
    title: str
    authors: UniqueList[str]

    contributors: UniqueList[str] = Field(default_factory=list)
    doi_is_draft: bool | None = True
    description: str | None = None
    publisher: str = "Garden-AI"
    year: str = Field(default_factory=lambda: str(datetime.now().year))
    language: str = "en"
    tags: UniqueList[str] = Field(default_factory=list)
    version: str = "0.0.1"
    entrypoint_aliases: dict[str, str] = Field(default_factory=dict)
    owner_identity_id: UUID | None = None
    entrypoint_ids: UniqueList[str] = Field(default_factory=list)
