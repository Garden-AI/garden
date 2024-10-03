from datetime import datetime
from uuid import UUID
from typing_extensions import Self
from pydantic import BaseModel, Field, model_validator

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
    """Represents the metadata defining a Garden, including the DOIs of its collected entrypoints.

    **Not meant to be instantiated directly by users.** Instead, new Gardens should be created via the CLI or [web UI](https://thegardens.ai/#/garden/create)

    Attributes:
        doi (str): The Digital Object Identifier (DOI) for the Garden.
        title (str): The title of the Garden.
        authors (UniqueList[str]): A list of authors of the Garden.
        contributors (UniqueList[str]): A list of contributors to the Garden. Defaults to an empty list.
        doi_is_draft (bool | None): Indicates if the DOI is in draft status. Defaults to True.
        description (str | None): A brief description of the Garden. Defaults to None.
        publisher (str): The publisher of the Garden. Defaults to "Garden-AI".
        year (str): The year of publication. Defaults to the current year.
        language (str): The primary language of the Garden. Defaults to "en" (English).
        tags (UniqueList[str]): A list of tags associated with the Garden. Defaults to an empty list.
        version (str): The version of the Garden. Defaults to "0.0.1".
        entrypoint_aliases (dict[str, str]): A dictionary mapping entrypoint DOIs to their aliases. Defaults to an empty dict.
        entrypoint_ids (UniqueList[str]): A list of entrypoint DOIs associated with this Garden. Defaults to an empty list.
        owner_identity_id (UUID | None): The UUID of the Garden's owner. Defaults to None.
        id (int | None): An internal identifier for the Garden. Defaults to None.
    """  # noqa: E501

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
    entrypoint_ids: UniqueList[str] = Field(default_factory=list)

    # TODO: for now a modal function ID is just the function's name, but once modal
    # functions have DOIs this should be updated for consistency with entrypoints.
    modal_function_ids: UniqueList[str] = Field(default_factory=list)

    owner_identity_id: UUID | None = None
    id: int | None = None

    @model_validator(mode="after")
    def _validate_aliases(self) -> Self:
        """Ensure aliases only refer to entrypoints actually in the garden."""
        known_dois = set(self.entrypoint_ids)
        aliased_dois = set(self.entrypoint_aliases.keys())
        for unknown_doi in aliased_dois - known_dois:
            del self.entrypoint_aliases[unknown_doi]
        return self

    def _datacite_json(self) -> JsonStr:
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
