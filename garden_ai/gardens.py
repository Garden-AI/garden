from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, cast
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ValidationError, validator

from garden_ai.utils.misc import JSON, garden_json_encoder

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


class PipelineNotFoundException(KeyError):
    """Exception raised when a Garden references an unknown pipeline uuid"""


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

    title: str = cast(str, Field(default_factory=lambda: None))
    authors: List[str] = Field(default_factory=list, min_items=1, unique_items=True)
    contributors: List[str] = Field(default_factory=list, unique_items=True)
    doi: Optional[str] = Field(default=None)
    description: Optional[str] = Field(None)
    publisher: str = "Garden"
    year: str = Field(default_factory=lambda: str(datetime.now().year))
    language: str = "en"
    tags: List[str] = Field(default_factory=list, unique_items=True)
    version: str = "0.0.1"
    uuid: UUID = Field(default_factory=uuid4, allow_mutation=False)
    pipeline_ids: List[UUID] = Field(default_factory=list)

    @validator("year")
    def valid_year(cls, year):
        if len(str(year)) != 4:
            raise ValueError("year must be formatted `YYYY`")
        return str(year)

    def collect_pipelines(self) -> List[RegisteredPipeline]:
        """Collect the full ``RegisteredPipeline`` objects referred to by this garden's pipeline_ids."""
        from .local_data import get_local_pipeline_by_uuid

        pipelines = []
        for uuid in self.pipeline_ids:
            pipeline = get_local_pipeline_by_uuid(uuid)
            if pipeline is None:
                raise PipelineNotFoundException(
                    f"Could not find registered pipeline with id {uuid}."
                )
            pipelines += [pipeline]
        return pipelines

    def expanded_metadata(self) -> Dict[str, Any]:
        """Helper: build the "complete" metadata dict with nested ``Pipeline`` and ``step`` metadata.

        Notes
        ------
        When serializing normally with ``garden.{dict(), json()}``, only the
        uuids of the pipelines in the garden are included.

        This returns a superset of ``garden.dict()``, so that the following holds:

            valid_garden == Garden(**valid_garden.expanded_metadata()) == Garden(**valid_garden.dict())

        Returns
        -------
        Dict[str, Any]  ``Garden`` metadata dict augmented with a list of ``RegisteredPipeline`` metadata

        Raises
        ------
        PipelineNotFoundException  if ``garden.pipeline_ids`` references an unknown pipeline id.
        """

        data = self.dict()
        data["pipelines"] = [p.dict() for p in self.collect_pipelines()]
        return data

    def expanded_json(self) -> JSON:
        """Helper: return the expanded garden metadata as JSON.

        See: ``Garden.expanded_metadata`` method
        """
        data = self.expanded_metadata()
        return json.dumps(data, default=garden_json_encoder)

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
                for p in self.collect_pipelines()
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
            _ = self.__class__(**self.dict())
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
        for pipe in self.collect_pipelines():
            new_contributors = set(pipe.authors) | set(pipe.contributors)
            known_contributors |= new_contributors - known_authors

        self.contributors = list(known_contributors)
        return
