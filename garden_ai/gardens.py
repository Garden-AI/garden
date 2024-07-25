from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)
from tabulate import tabulate
from typing_extensions import Annotated

from garden_ai.schemas.datacite import (
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
from garden_ai.schemas.garden import GardenMetadata
from garden_ai.utils.misc import JSON
from garden_ai.utils.pydantic import unique_items_validator

from .entrypoints import Entrypoint_, RegisteredEntrypoint

logger = logging.getLogger()
require_unique_items = AfterValidator(unique_items_validator)


class Garden_:
    def __init__(self, metadata: GardenMetadata, entrypoints: list[Entrypoint_]):
        if set(metadata.entrypoint_ids) != set([ep.metadata.doi for ep in entrypoints]):
            raise ValueError(
                "Expected `entrypoints` DOIs to match `metadata.entrypoint_ids`. "
                f"Got: {[ep.metadata.doi for ep in entrypoints]} != {metadata.entrypoint_ids}"
            )
        self.metadata = metadata
        self.entrypoints = entrypoints
        return

    def __getattr__(self, name):
        # enables method-like syntax for calling entrypoints from this garden.
        # note: this is only called as a fallback when __getattribute__ raises an exception,
        # existing attributes are not affected by overriding this
        message_extra = ""
        for entrypoint in self.entrypoints:
            short_name = entrypoint.metadata.short_name
            alias = (
                self.metadata.entrypoint_aliases.get(entrypoint.metadata.doi)
                or short_name
            )
            if name == alias:
                return entrypoint
            elif name == short_name:
                message_extra = f" Did you mean {alias}?"

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'."
            + message_extra
        )

    def __dir__(self):
        # this gets us jupyter/ipython/repl tab-completion of entrypoint names
        entrypoint_names = []
        for entrypoint in self.entrypoints:
            name = self.metadata.entrypoint_aliases.get(entrypoint.metadata.doi)
            if name is None:
                name = entrypoint.metadata.short_name
            entrypoint_names += [name]

        return list(super().__dir__()) + entrypoint_names

    def _repr_html_(self) -> str:
        data = self.metadata.model_dump()
        data["entrypoints"] = [ep.metadata.model_dump() for ep in self.entrypoints]

        style = "<style>th {text-align: left;}</style>"
        title = f"<h2>{data['title']}</h2>"
        details = f"<p>Authors: {', '.join(data['authors'])}<br>DOI: {data['doi']}</p>"
        entrypoints = "<h3>Entrypoints</h3>" + tabulate(
            [
                {
                    key.title(): str(entrypoint[key])
                    for key in ("short_name", "title", "authors", "doi")
                }
                for entrypoint in data["entrypoints"]
            ],
            headers="keys",
            tablefmt="html",
        )
        optional = "<h3>Additional data</h3>" + tabulate(
            [
                (field, str(val))
                for field, val in data.items()
                if field not in ("title", "authors", "doi", "short_name")
                and "entrypoint" not in field
                and val
            ],
            tablefmt="html",
        )
        return style + title + details + entrypoints + optional


class PublishedGarden(BaseModel):
    """Metadata of a completed and published `Garden` object.

    Attributes:
        authors (list[str]):
            The main researchers involved in producing the Garden. \
            Personal name format should be: "Family, Given".

        contributors (list[str]):
            Acknowledge contributors to the development of this Garden. These\
            should be distinct from `authors`.

        title (str):
            An official name or title for the Garden.

        description (str):
            A brief summary of the Garden and/or its purpose, to aid discovery by \
            other Gardeners.

        entrypoints (list[RegisteredEntrypoint]):
            List of the entrypoints associated with this garden \
            Note that these entrypoints can also be accessed directly by their \
            `short_name` (or alias).

        entrypoint_aliases: Dict[str, str] = Field(default_factory=dict)

        doi (str):
            A garden's DOI usable for citations.

        version (str):
            optional, defaults to "0.0.1".

        language (str):
            Recommended values are IETF BCP 47, ISO 639-1 language codes, such as:\
            "en", "de", "fr". Defaults to "en".

        tags (List[str]):
            Tags, keywords, classification codes, or key phrases pertaining to the Garden.

        year (str):
            Defaults to current year. This attribute should be formatted 'YYYY'.

    Examples:
        Retrieving a remote Garden with a ``GardenClient`` instance::

        ```python
        gc = GardenClient()
        garden = gc.get_published_garden("10.23677/placeholder")
        ```
    """

    model_config = ConfigDict(validate_default=False, validate_assignment=True)

    title: str = Field(...)
    authors: List[str] = Field(...)
    contributors: Annotated[
        List[str], Field(default_factory=list), require_unique_items
    ]
    doi: str = Field(...)
    doi_is_draft: bool = Field(True)
    description: Optional[str] = Field(None)
    publisher: str = "Garden-AI"
    year: str = Field(default_factory=lambda: str(datetime.now().year))
    language: str = "en"
    tags: Annotated[List[str], Field(default_factory=list), require_unique_items]
    version: str = "0.0.1"
    entrypoints: List[RegisteredEntrypoint] = Field(...)
    entrypoint_aliases: Dict[str, str] = Field(default_factory=dict)

    @field_validator("year")
    @classmethod
    def valid_year(cls, year):
        if len(str(year)) != 4:
            raise ValueError("year must be formatted `YYYY`")
        return str(year)

    @classmethod
    def from_garden(cls, garden) -> PublishedGarden:
        """Helper: instantiate a `PublishedGarden` directly from a `Garden` instance.

        Raises:
            ValidationError:
                If any fields required by `PublishedGarden` but not \
                `Garden` are not set.
        """
        # note: we want every PublishedGarden to be re-constructible
        # from mere json, so as a sanity check we use garden.json() instead of
        # garden.dict() directly
        record = garden.expanded_json()
        data = json.loads(record)
        return cls(**data)

    def _repr_html_(self) -> str:
        return garden_repr_html(self)

    def datacite_json(self) -> JSON:
        """Parse this `Garden`s metadata into a DataCite-schema-compliant JSON string."""

        # Leverages a pydantic class `DataCiteSchema`, which was automatically generated from:
        # https://github.com/datacite/schema/blob/master/source/json/kernel-4.3/datacite_4.3_schema.json
        #
        # The JSON returned by this method would be the "attributes" part of a DataCite request body.

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
                for doi in (p.doi for p in self.entrypoints)
            ],
            version=self.version,
            descriptions=(
                [Description(description=self.description, descriptionType="Other")]
                if self.description
                else None
            ),
        ).model_dump_json()

    def __getattr__(self, name):
        #  note: this is only called as a fallback when __getattribute__ raises an exception,
        #  existing attributes are not affected by overriding this
        message_extra = ""
        for entrypoint in self.entrypoints:
            short_name = entrypoint.short_name
            alias = self.entrypoint_aliases.get(entrypoint.doi) or short_name
            if name == alias:
                return entrypoint
            elif name == short_name:
                message_extra = f" Did you mean {alias}?"

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'."
            + message_extra
        )

    def __dir__(self):
        # this gets us jupyter/ipython/repl tab-completion of entrypoint names
        entrypoint_names = [
            self.entrypoint_aliases.get(p.doi) or p.short_name for p in self.entrypoints
        ]
        return list(super().__dir__()) + entrypoint_names


def garden_repr_html(garden: PublishedGarden) -> str:
    data = garden.dict()

    style = "<style>th {text-align: left;}</style>"
    title = f"<h2>{data['title']}</h2>"
    details = f"<p>Authors: {', '.join(data['authors'])}<br>DOI: {data['doi']}</p>"
    entrypoints = "<h3>Entrypoints</h3>" + tabulate(
        [
            {key.title(): str(entrypoint[key]) for key in ("title", "authors", "doi")}
            for entrypoint in data["entrypoints"]
        ],
        headers="keys",
        tablefmt="html",
    )
    optional = "<h3>Additional data</h3>" + tabulate(
        [
            (field, str(val))
            for field, val in data.items()
            if field not in ("title", "authors", "doi")
            and "entrypoint" not in field
            and val
        ],
        tablefmt="html",
    )
    return style + title + details + entrypoints + optional
