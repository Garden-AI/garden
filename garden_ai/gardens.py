from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, cast
from typing_extensions import Annotated

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    model_validator,
    field_validator,
    AfterValidator,
    ConfigDict,
)
from pydantic_core import to_jsonable_python
from tabulate import tabulate

from garden_ai.utils.misc import JSON
from garden_ai.utils.pydantic import unique_items_validator

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
from .entrypoints import RegisteredEntrypoint

logger = logging.getLogger()
require_unique_items = AfterValidator(unique_items_validator)


class Garden(BaseModel):
    """Object representation of a Garden.

    Attributes:
        authors (list[str]):
            The main researchers involved in producing the Garden. At least one \
            author is required in order to register a DOI. Personal name format \
            should be: "Family, Given".

        contributors (list[str]):
            Acknowledge contributors to the development of this Garden. These\
            should be distinct from `authors`.

        title (str):
            An official name or title for the Garden. This attribute must be set \
            in order to register a DOI.

        description (str):
            A brief summary of the Garden and/or its purpose, to aid discovery by \
            other Gardeners.

        entrypoint_ids: List[str] = Field(default_factory=list)
        entrypoint_aliases: Dict[str, str] = Field(default_factory=dict)

        doi (str):
            A garden's DOI usable for citations, generated automatically via \
            DataCite using the required fields.

        doi_is_draft: (bool):
            Used to track the status of this garden's DOI. All DOIs start out as drafts. \
            You can promote a DOI to a registered state with `garden-ai garden register-doi`.

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
        Creating a new Garden with a ``GardenClient`` instance::

        ```python
        gc = GardenClient()
        pea_garden = gc.create_garden(
            authors=["Mendel, Gregor"], title="Experiments on Plant Hybridization"
        )
        pea_garden.year = '1863'
        pea_garden.description = '''This Garden houses sophisticated ML entrypoints
                                    for Big Pea Data extraction and classification.
                                    It consists of a 2-hectare plot behind the monastery,
                                    and a 30,000-plant dataset.'''
        ```
    Notes:
        Mendel's work was ignored by the scientific community during his lifetime, \
        presumably due to the lack of a working DOI. \
        To remedy this, if the `doi` field is unset when publishing a garden, we \
        build one for the user with the datacite api.
    """

    model_config = ConfigDict(validate_default=False, validate_assignment=True)

    title: str = cast(str, Field(default_factory=lambda: None))
    authors: Annotated[
        List[str], Field(default_factory=list, min_length=1), require_unique_items
    ]
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
    entrypoint_ids: List[str] = Field(default_factory=list)
    entrypoint_aliases: Dict[str, str] = Field(default_factory=dict)
    _entrypoint_cache: List[RegisteredEntrypoint] = PrivateAttr(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def doi_omitted(cls, values):
        assert "doi" in values, (
            "It seems like no DOI has been minted yet for this `Garden`. If you were trying to create a new `Garden`, "
            "use `GardenClient.create_garden` to initialize a publishable `Garden` with a draft DOI."
        )
        return values

    @field_validator("year")
    @classmethod
    def valid_year(cls, year):
        if len(str(year)) != 4:
            raise ValueError("year must be formatted `YYYY`")
        return str(year)

    def __init__(
        self, _entrypoints: Optional[List[RegisteredEntrypoint]] = None, **kwargs
    ):
        super().__init__(**kwargs)

        if _entrypoints is not None:
            # HACK: for compatibility with usage in app.garden._get_garden
            # skips expensive _collect_entrypoints() call when we already have
            # them from the GET /gardens response
            self._entrypoint_cache = _entrypoints
            return

        self._entrypoint_cache = self._collect_entrypoints()
        return

    def _repr_html_(self) -> str:
        return garden_repr_html(self)

    def _collect_entrypoints(self) -> List[RegisteredEntrypoint]:
        """
        Collects the entrypoint objects that have been registered to this garden from the local database.

        Returns:
            A list of RegisteredEntrypoint objects.
        """

        # HACK: this is ugly, but no more so than the circular
        # local_data import it is replacing. As soon as the migration is
        # complete and we can remove the local_data module itself we can remove
        # the affected methods too (maybe even removing the whole Garden class).
        # This is the compromise while this is still a load-bearing helper method.
        import garden_ai.local_data

        if garden_ai.local_data._IS_DISABLED:
            entrypoints = []
            from garden_ai.client import GardenClient

            client = GardenClient()
            for doi in self.entrypoint_ids:
                entrypoint = client.backend_client.get_entrypoint(doi)
                entrypoints += [entrypoint]
            return entrypoints

        from .local_data import EntrypointNotFoundException, get_local_entrypoint_by_doi

        entrypoints = []
        for doi in self.entrypoint_ids:
            entrypoint = get_local_entrypoint_by_doi(doi)  # type: ignore[assignment]
            if entrypoint is None:
                raise EntrypointNotFoundException(
                    f"Could not find registered entrypoint with id {doi}."
                )

            entrypoints += [entrypoint]

        return entrypoints

    def add_entrypoint(
        self, entrypoint_id: str, alias: Optional[str] = None, replace=False
    ):
        """
        Fetches the entrypoint with the given DOI from the local database and adds it to this garden.

        If replace=True, any entrypoint with the same `short_name` will be
            replaced, else raise a ValueError (default)

        Raises:
            ValueError: If any of the provided arguments would result in an invalid state.
        """
        if entrypoint_id in self.entrypoint_ids:
            if replace:
                self.remove_entrypoint(entrypoint_id)
                self._entrypoint_cache = self._collect_entrypoints()
            else:
                raise ValueError(
                    "Error: this entrypoint is already attached to this garden. "
                    "to rename an entrypoint, see `rename_entrypoint`"
                )

        import garden_ai.local_data

        # HACK: same workaround as _collect_entrypoints. This method will be safe to remove
        # once we're in a post-local_data-world.
        if garden_ai.local_data._IS_DISABLED:
            from garden_ai.client import GardenClient

            client = GardenClient()
            entrypoint = client.backend_client.get_entrypoint(entrypoint_id)

        else:
            from .local_data import get_local_entrypoint_by_doi

            entrypoint = get_local_entrypoint_by_doi(entrypoint_id)  # type: ignore[assignment]
            if entrypoint is None:
                raise ValueError(
                    f"Error: no entrypoint was found in the local database with the given DOI {entrypoint_id}."
                )

        entrypoint_names = (
            self.entrypoint_aliases.get(cached.doi) or cached.short_name
            for cached in self._entrypoint_cache
        )
        if alias is None and entrypoint.short_name in entrypoint_names:
            raise ValueError(
                f"Error: an entrypoint with the name {entrypoint.short_name} already exists in this garden, "
                "please provide an alias for the new entrypoint."
            )

        self.entrypoint_ids += [entrypoint_id]
        self._entrypoint_cache += [entrypoint]

        if alias:
            self.rename_entrypoint(entrypoint_id, alias)
        return

    def remove_entrypoint(self, entrypoint_id: str):
        if entrypoint_id not in self.entrypoint_ids:
            raise ValueError(
                f"Error: no entrypoint with DOI {entrypoint_id} found in this garden. "
            )
        self.entrypoint_ids.remove(entrypoint_id)
        if entrypoint_id in self.entrypoint_aliases:
            del self.entrypoint_aliases[entrypoint_id]

        return

    def expanded_metadata(self) -> Dict[str, Any]:
        """
        Helper method: builds the "complete" metadata dictionary with nested `Entrypoint` and `step` metadata.

        When serializing normally with `garden.{dict(), json()}`, only the DOIs of the entrypoints in the garden are included.

        This method returns a superset of `garden.dict()`, so that the following holds:
            valid_garden == Garden(**valid_garden.expanded_metadata()) == Garden(**valid_garden.dict())

        Returns:
            A dictionary containing the `Garden` metadata augmented with a list of `RegisteredEntrypoint` metadata.

        Raises:
            EntrypointNotFoundException: If `garden.entrypoint_ids` references an unknown entrypoint ID.
        """
        self._sync_author_metadata()
        data = self.model_dump()
        data["entrypoints"] = [p.model_dump() for p in self._collect_entrypoints()]
        return data

    def expanded_json(self) -> JSON:
        """Helper: return the expanded garden metadata as JSON.

        See: ``Garden.expanded_metadata`` method
        """
        data = self.expanded_metadata()
        return json.dumps(data, default=to_jsonable_python)

    def _sync_author_metadata(self):
        known_contributors = set(self.contributors)
        # garden contributors don't need to duplicate garden authors unless they've been explicitly added
        known_authors = set(self.authors) - known_contributors

        for entrypoint in self._entrypoint_cache:
            new_contributors = set(entrypoint.authors)
            known_contributors |= new_contributors - known_authors

        self.contributors = list(known_contributors)
        return

    def rename_entrypoint(self, entrypoint_id: str, new_name: str):
        """Rename an entrypoint in this garden.

        Parameters:
            entrypoint_id (str): the DOI for the entrypoint to be renamed
            new_name (str): the new short_name of the entrypoint
        Raises:
            ValueError: if the new_name is already in use, or if the old_name is \
            not found, or if the new_name is not a valid identifier.
        """
        if not new_name.isidentifier():
            raise ValueError("an alias must be a valid Python identifier.")

        if entrypoint_id not in self.entrypoint_ids:
            raise ValueError(
                f"Error: could not find entrypoint with DOI {entrypoint_id} in this garden."
            )

        entrypoint_names = (
            self.entrypoint_aliases.get(cached.doi) or cached.short_name
            for cached in self._entrypoint_cache
        )
        if new_name in entrypoint_names:
            raise ValueError(
                f"Error: found existing entrypoint with name {new_name} in this garden."
            )

        self.entrypoint_aliases[entrypoint_id] = new_name
        return


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
    def from_garden(cls, garden: Garden) -> PublishedGarden:
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


def garden_repr_html(garden: Union[Garden, PublishedGarden]) -> str:
    if isinstance(garden, Garden):
        data = garden.expanded_metadata()
    else:
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
