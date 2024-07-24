from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from .datacite import (
    Contributor,
    Creator,
    DataciteSchema,
    Description,
    Identifier,
    RelatedIdentifier,
    Subject,
    Title,
    Types,
)
from .schema_utils import JsonStr, UniqueList


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

    def datacite_json(self) -> JsonStr:
        """Convert metadata into a DataCite-schema-compliant JSON string."""
        return DataciteSchema(  # type: ignore
            identifiers=[Identifier(identifier=self.doi, identifierType="DOI")],
            types=Types(resourceType="AI/ML Garden", resourceTypeGeneral="Software"),
            creators=[Creator(name=name) for name in self.authors],
            titles=[Title(title=self.title)],
            publisher="thegardens.ai",
            publicationYear=self.year,
            subjects=[Subject(subject=tag) for tag in self.tags],
            contributors=[
                Contributor(name=name, contributorType="Other")
                for name in self.contributors
            ],
            language=self.language,
            relatedIdentifiers=[
                RelatedIdentifier(
                    relatedIdentifier=doi,
                    relatedIdentifierType="DOI",
                    relationType="HasPart",
                )
                for doi in self.entrypoint_ids
            ],
            version=self.version,
            descriptions=(
                [Description(description=self.description, descriptionType="Other")]
                if self.description
                else None
            ),
        ).model_dump_json()
