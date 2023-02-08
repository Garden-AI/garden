from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional, cast
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ValidationError, validator

from .datacite import (
    Contributor,
    Creator,
    DataciteSchema,
    Description,
    RelatedIdentifier,
    Title,
    Types,
)
from .utils import garden_json_encoder
from .pipelines import Pipeline
from .steps import Step

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
    build one for the user with the datacite api (see the `request_doi()`
    method).
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
    version: str = "0.0.1"  # TODO: enforce semver for this?
    pipelines: List[Pipeline] = Field(default_factory=list)
    garden_id: UUID = Field(default_factory=uuid4, allow_mutation=False)

    @validator("year")
    def valid_year(cls, year):
        if len(str(year)) != 4:
            raise ValueError("year must be formatted `YYYY`")
        return str(year)

    def json(self, **kwargs):
        kwargs.update(encoder=garden_json_encoder)
        return super().json(**kwargs)

    def datacite_json(self) -> str:
        """Parse this `Garden`s metadata into a DataCite-schema-compliant JSON string.

        Leverages a pydantic class `DataCiteSchema`, which was automatically generated from:
        https://github.com/datacite/schema/blob/master/source/json/kernel-4.3/datacite_4.3_schema.json

        The JSON returned by this method would be the "attributes" part of a DataCite request body.
        """
        self._sync_author_metadata()
        return DataciteSchema(
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
            pipe._sync_author_metadata()
            new_contributors = set(pipe.authors) | set(pipe.contributors)
            known_contributors |= new_contributors - known_authors

        self.contributors = list(known_contributors)
        return

    def add_new_pipeline(
        self, title: str, steps: List[Step], authors: List[str] = None, **kwargs
    ):
        """Create a new Pipeline object and add it to this Garden's list of pipelines.

        Arguments (along with any further `kwargs`, e.g. `description`) have the
        same meaning as in `Pipeline` constructor.

        If not provided, the `authors` field of the pipeline will be set to this
        garden's `authors` attribute.

        """
        kwargs.update(
            authors=authors if authors else self.authors,
            title=title,
            steps=tuple(steps),
        )
        pipe = Pipeline(**kwargs)
        self.pipelines += [pipe]
        return

    def remove_pipeline(self, to_remove: Pipeline):
        """Removes a given Pipeline object from this Garden, if present."""
        self.pipelines = [p for p in self.pipelines if p.uuid != to_remove.uuid]
        return
