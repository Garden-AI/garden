from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional, cast
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ValidationError, root_validator, validator

from garden_ai.utils.misc import JSON

from .datacite import (
    Contributor,
    Creator,
    DataciteSchema,
    Description,
    RelatedIdentifier,
    Title,
    Types,
)
from .pipelines import RegisteredPipeline

logger = logging.getLogger()


class Garden(BaseModel):
    """Object representation of a Garden™.

    Required Attributes
    --------------------
    authors: List[str]
        The main researchers involved in producing the Garden. At least one
        author is required in order to register a DOI. Personal name format
        should be: "Family, Given".

    title: str
        An official name or title for the Garden. This attribute must be set
        in order to register a DOI.


    Recommended Attributes
    --------------------
    description: str
        A brief summary of the Garden and/or its purpose, to aid discovery by
        other Gardeners.

    pipelines: List[Pipeline]
        TODO

    Optional Attributes
    --------------------
    doi: str
        A garden's doi can be manually set by the user, or generated automatically via the
        DataCite rest api using the required fields.

    language: str
        Recommended values are IETF BCP 47, ISO 639-1 language codes, such as:
        "en", "de", "fr". Defaults to "en".

    tags: List[str]
        Tags, keywords, classification codes, or key phrases pertaining to the Garden.

    year: str
        Defaults to current year. This attribute should be formatted 'YYYY'.

    Examples
    --------
    Creating a new Garden with a ``GardenClient`` instance::

        gc = GardenClient()
        pea_garden = gc.create_garden(
            authors=["Mendel, Gregor"], title="Experiments on Plant Hybridization"
        )
        pea_garden.year = 1863
        pea_garden.description = '''This Garden houses sophisticated ML pipelines
                                  for Big Pea Data extraction and classification.
                                  It consists of a 2-hectare plot behind the monastery,
                                  and a 30,000-plant dataset.'''

        gc.register(pea_garden)

    Notes
    --------
    Mendel's work was ignored by the scientific community during his lifetime,
    presumably due to the lack of a working DOI.
    To remedy this, if the `doi` field is unset when registering the garden, we
    build one for the user with the datacite api.
    """

    class Config:
        """
        Configure pydantic per-model settings.

        We disable validate_all so that pydantic ignores temporarily-illegal defaults
        We enable validate_assignment to make validation occur naturally even after object construction
        """

        validate_all = False  # (this is the default)
        validate_assignment = True  # validators invoked on assignment
        underscore_attrs_are_private = True

    authors: List[str] = Field(default_factory=list, min_items=1, unique_items=True)
    contributors: List[str] = Field(default_factory=list, unique_items=True)
    # note: default_factory=lambda:None allows us to have fields which are None by
    # default, but not automatically considered optional by pydantic
    title: str = cast(str, Field(default_factory=lambda: None))
    doi: str = cast(str, Field(default_factory=lambda: None))
    # ^ casts here to appease mypy
    description: Optional[str] = Field(None)
    publisher: str = "Garden"
    year: str = Field(default_factory=lambda: str(datetime.now().year))
    language: str = "en"
    tags: List[str] = Field(default_factory=list, unique_items=True)
    version: str = "0.0.1"
    uuid: UUID = Field(default_factory=uuid4, allow_mutation=False)
    pipelines: List[RegisteredPipeline] = Field(
        default_factory=list, include={"__all__": {"uuid"}}
    )  # see: fetch_registered_pipeline below

    @validator("year")
    def valid_year(cls, year):
        if len(str(year)) != 4:
            raise ValueError("year must be formatted `YYYY`")
        return str(year)

    @root_validator(pre=True)
    def fetch_registered_pipelines(cls, values):
        if "pipelines" not in values:
            return values
        from .local_data import get_local_pipeline_by_uuid

        # HACK - because we tell pydantic to only include 'uuid' from
        # pipelines when dumping Garden json/dict, we break the invariant of
        # `valid_garden == Garden(**valid_garden.dict())` unless we intercept
        # the bare uuids here and replace them with the real thing.

        pipelines = []
        for p in values["pipelines"]:
            if isinstance(p, dict):
                pipelines += [get_local_pipeline_by_uuid(p["uuid"])]
            else:
                pipelines += [p]
        values["pipelines"] = pipelines
        return values

    def datacite_json(self) -> JSON:
        """Parse this `Garden`s metadata into a DataCite-schema-compliant JSON string.

        Leverages a pydantic class `DataCiteSchema`, which was automatically generated from:
        https://github.com/datacite/schema/blob/master/source/json/kernel-4.3/datacite_4.3_schema.json

        The JSON returned by this method would be the "attributes" part of a DataCite request body.
        """
        self._sync_author_metadata()
        return DataciteSchema(  # type: ignore
            types=Types(resourceType="AI/ML Garden", resourceTypeGeneral="Software"),
            creators=[Creator(name=name) for name in self.authors],
            titles=[Title(title=self.title)],
            publisher="thegardens.ai",
            publicationYear=self.year,
            contributors=[
                Contributor(name=name, contributorType="Other")
                for name in self.contributors
            ],
            language=self.language,
            relatedIdentifiers=[
                RelatedIdentifier(
                    relatedIdentifier=p.doi,
                    relatedIdentifierType="DOI",
                    relationType="HasPart",
                )
                for p in self.pipelines
                if p.doi
            ],
            version=self.version,
            descriptions=[
                Description(description=self.description, descriptionType="Other")
            ]
            if self.description
            else None,
        ).json()

    def validate(self):
        """Perform validation on all fields, even fields which are still defaults.

        This is useful for catching fields with a default value which would not
        otherwise be valid, (e.g.  `None` for any required field), in particular
        as a sanity check before committing to publishing/registering the user's
        Garden.

        This is mostly redundant for any already-set fields, as the validators
        for those would have (hopefully) already run when they were set. However
        (because this is still python), it's relatively easy to *modify* some
        fields without ever *assigning* to them, which wouldn't trigger
        validation.

        Examples
        --------
        gc = GardenClient()
        pea_garden = gc.create_garden(
            authors=["Mendel, Gregor"], title="Experiments on Plant Hybridization"
        )

        # NOTE: pydantic won't see this, as it's neither construction nor assignment
        pea_garden.authors.append(None)         # no complaints

        # checks all fields, even those smuggled in without triggering validation.
        pea_garden.validate()
        # ...
        # ValidationError: 1 validation error for Garden
        # authors -> 1
        #   none is not an allowed value (type=type_error.none.not_allowed)
        """
        try:
            self._sync_author_metadata()
            _ = self.__init__(**self.dict())
        except ValidationError as err:
            logger.error(err)
            raise

    def _sync_author_metadata(self):
        """helper: authors and contributors of steps and Pipelines also appear as contributors in their respective Pipeline and Garden's metadata.

        We'll probably want to tweak the behavior of this at some point once we
        tease out what we really want `contributor` to entail, but for now this
        seems like a sane default.
        """
        known_contributors = set(self.contributors)
        # garden contributors don't need to duplicate garden authors unless they've been explicitly added
        known_authors = set(self.authors) - known_contributors
        for pipe in self.pipelines:
            new_contributors = set(pipe.authors) | set(pipe.contributors)
            known_contributors |= new_contributors - known_authors

        self.contributors = list(known_contributors)
        return
