#!/usr/bin/env python3

from __future__ import annotations
import logging
from datetime import datetime
from typing import List
from uuid import uuid4, UUID
from pydantic import BaseModel, Field, validator, ValidationError

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


    Optional Attributes
    --------------------
    doi: str
        A garden's doi can be manually set by the user, or generated automatically via the
        DataCite rest api using the required fields.

    language: str
        Recommended values are IETF BCP 47, ISO 639-1 language codes, such as:
        "en", "de", "fr". Defaults to "en".

    subjects: List[str]
        Subjects, keywords, classification codes, or key phrases pertaining to the Garden.

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
    To remedy this, if the doi field is unset when registering the
    garden, we could just build one for the user with the datacite api.
    This could also eventually be exposed as a `register_doi()` method.
    """

    #
    __required__: List[str] = ["authors", "title", "doi"]
    __recommended__: List[str] = ["description", "subjects", "version"]

    # fields required for the DataCite rest api to generate a findable DOI
    __doi_required__: List[str] = [
        "_doi_prefix",
        "authors",
        "title",
        "publisher",
        "year",
        "resourceTypeGeneral",
    ]
    _doi_prefix = "10.26311"

    class Config:
        """
        Configure pydantic per-model settings.

        We disable validate_all so that pydantic ignores temporarily-illegal defaults
        We enable validate_assignment to make validation occur naturally even after object construction
        """

        validate_all = False  # (this is the default)
        validate_assignment = True  # validators invoked (only?) on assignment

    authors: List[str] = Field(default_factory=list, min_items=1, unique_items=True)
    title: str = Field(default_factory=lambda: None)
    resourceTypeGeneral: str = "Other"  # (or: model, software, service, interactive?)
    publisher: str = "Garden"
    year: str = Field(default_factory=lambda: str(datetime.now().year))

    doi: str = Field(default_factory=lambda: None)

    language: str = "en"
    subjects: List[str] = Field(default_factory=list, unique_items=True)
    description: str = Field(None)
    version: str = "0.0.1"  # TODO: enforce semver for this?

    # field(s) for which we might want to ''disable'' mutation
    garden_id: UUID = Field(default_factory=uuid4, allow_mutation=False)

    @validator("authors", each_item=True)
    def valid_name(cls, author_name: str):
        """''Validate''a single `author` string by returning it unchanged.

        This currently might even do *less* than pydantic would have otherwise
        done for the field, but this is probably where we'd want to put the logic to
        handle input strings as authors vs as institutions/etc as input.

        Parameters
        ----------
        cls : Garden
            Garden instance whose `authors` attribute has been modified
        author_name : str
            single author name to validate (not list of authors)

        Raises
        ---------
        This should (eventually) raise a `ValueError`, `TypeError`, or `AssertionError`, per pydantic docs
        """

        # invoked per-author, not list of authors
        # todo: validate with name parser lib?
        # todo: institution vs personal name?
        return str(author_name)

    @validator("year")
    def valid_year(cls, year):
        if len(str(year)) != 4:
            raise ValueError("year must be formatted `YYYY`")
        return str(year)

    def request_doi(self):
        if self.doi:
            return self.doi
        for name in self.__doi_required__:
            if not self.__getattribute__(name):
                logger.error(
                    f"{name} is required to register a new doi, but has not been set."
                )
                return
        # TODO this should eventuelly hit the datacite api

        self.doi = self._doi_prefix + "/fake-doi"
        return self.doi

    def to_do(self):
        """Log errors and warnings for unset required and recommended fields, respectively.

        Does not raise any exceptions, unlike `validate()`.

        I think it seems useful to have a friendlier way to inform the user
        about missing fields with more granularity than `validate()`, which
        doesn't care about our not-required-but-recommended distinctions.

        This is a proof-of-concept convenience function as much as anything, and
        I would't be surprised if we move this behavior somewhere else or decide
        it's redundant.
        """

        for name in self.__required__:
            if not self.__getattribute__(name):
                logger.error(f"{name} is a required attribute, but has not been set.")
        for name in self.__recommended__:
            if not self.__getattribute__(name):
                logger.warning(
                    f"{name} is not a required attribute, but is strongly recommended and has not been set."
                )

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
        MC_LOVIN = None         # (clearly, this is not a valid name)
        pea_garden.authors.append(MCLOVIN)

        # checks all fields, even those smuggled in without triggering validation.
        pea_garden.validate()
        # ...
        # ValidationError: 1 validation error for Garden
        # authors -> 1
        #   none is not an allowed value (type=type_error.none.not_allowed)
        """
        try:
            _ = self.__init__(**self.dict())
        except ValidationError as err:
            logger.error(err)
            raise
