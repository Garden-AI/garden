from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, cast

from tabulate import tabulate
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    ValidationError,
    root_validator,
    validator,
)

from garden_ai.utils.misc import JSON, garden_json_encoder

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
from .pipelines import RegisteredPipeline

logger = logging.getLogger()


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

        pipelines (list[RegisteredPipeline]):
            List of the pipelines associated with this garden, built dynamically \
            from the `pipeline_ids` attribute. Note that these pipelines can \
            also be accessed directly by their `short_name` (or alias) - see \
            also `pipeline_names` attribute.

        pipeline_names (list[str]):
            List of python identifiers (i.e. variable names) usable for this \
            garden's pipelines. Takes aliases into account (set when adding \
            pipeline via CLI or using `rename_pipeline` method.)

        pipeline_ids: List[str] = Field(default_factory=list)
        pipeline_aliases: Dict[str, str] = Field(default_factory=dict)

        doi (str):
            A garden's doi usable for citations, generated automatically via \
            DataCite using the required fields.

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
        pea_garden.description = '''This Garden houses sophisticated ML pipelines
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

    title: str = cast(str, Field(default_factory=lambda: None))
    authors: List[str] = Field(default_factory=list, min_items=1, unique_items=True)
    contributors: List[str] = Field(default_factory=list, unique_items=True)
    doi: str = Field(...)
    description: Optional[str] = Field(None)
    publisher: str = "Garden-AI"
    year: str = Field(default_factory=lambda: str(datetime.now().year))
    language: str = "en"
    tags: List[str] = Field(default_factory=list, unique_items=True)
    version: str = "0.0.1"
    pipelines: List[RegisteredPipeline] = Field(default_factory=list)
    pipeline_ids: List[str] = Field(default_factory=list)
    pipeline_names: List[str] = Field(default_factory=list)
    pipeline_aliases: Dict[str, str] = Field(default_factory=dict)
    _env_vars: Dict[str, str] = PrivateAttr(default_factory=dict)

    @root_validator(pre=True)
    def doi_omitted(cls, values):
        assert "doi" in values, (
            "It seems like no DOI has been minted yet for this `Garden`. If you were trying to create a new `Garden`, "
            "use `GardenClient.create_garden` to initialize a publishable `Garden` with a draft DOI."
        )
        return values

    @validator("year")
    def valid_year(cls, year):
        if len(str(year)) != 4:
            raise ValueError("year must be formatted `YYYY`")
        return str(year)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pipelines = self.pipelines or self._collect_pipelines()
        self.pipeline_names = self.pipeline_names or [
            self.pipeline_aliases.get(pipeline.short_name) or pipeline.short_name
            for pipeline in self.pipelines
        ]
        return

    def _repr_html_(self) -> str:
        return garden_repr_html(self)

    def _collect_pipelines(self) -> List[RegisteredPipeline]:
        """
        Collects the pipeline objects that have been registered to this garden from the local database.

        Returns:
            A list of RegisteredPipeline objects.
        """
        from .local_data import PipelineNotFoundException, get_local_pipeline_by_doi

        pipelines = []
        for doi in self.pipeline_ids:
            pipeline = get_local_pipeline_by_doi(doi)
            if pipeline is None:
                raise PipelineNotFoundException(
                    f"Could not find registered pipeline with id {doi}."
                )
            # set env vars for pipeline to use when remotely executing
            pipeline._env_vars = self._env_vars

            pipelines += [pipeline]

        return pipelines

    def add_pipeline(self, pipeline_id: str, alias: Optional[str] = None):
        """
        Fetches the pipeline with the given DOI from the local database and adds it to this garden.

        Raises:
            ValueError: If any of the provided arguments would result in an invalid state.
        """
        from .local_data import get_local_pipeline_by_doi

        pipeline = get_local_pipeline_by_doi(pipeline_id)
        if pipeline is None:
            raise ValueError(
                f"Error: no pipeline was found in the local database with the given doi {pipeline_id}."
            )

        if pipeline.short_name in self.pipeline_names and alias is None:
            raise ValueError(
                f"Error: a pipeline with the name {pipeline.short_name} already exists in this garden, "
                "please provide an alias for the new pipeline."
            )

        self.pipelines.append(pipeline)
        self.pipeline_ids.append(pipeline_id)
        self.pipeline_names.append(pipeline.short_name)

        if alias:
            try:
                self.rename_pipeline(pipeline.short_name, alias)
            except ValueError as e:
                # re-raise this particular exception so that the correct parameter is listed
                if "valid identifier" in str(e):
                    raise ValueError("alias must be a valid identifier.") from e
                raise
        return

    def expanded_metadata(self) -> Dict[str, Any]:
        """
        Helper method: builds the "complete" metadata dictionary with nested `Pipeline` and `step` metadata.

        When serializing normally with `garden.{dict(), json()}`, only the DOIs of the pipelines in the garden are included.

        This method returns a superset of `garden.dict()`, so that the following holds:
            valid_garden == Garden(**valid_garden.expanded_metadata()) == Garden(**valid_garden.dict())

        Returns:
            A dictionary containing the `Garden` metadata augmented with a list of `RegisteredPipeline` metadata.

        Raises:
            PipelineNotFoundException: If `garden.pipeline_ids` references an unknown pipeline ID.
        """
        self._sync_author_metadata()
        data = self.dict()
        data["pipelines"] = [p.expanded_metadata() for p in self.pipelines]
        return data

    def expanded_json(self) -> JSON:
        """Helper: return the expanded garden metadata as JSON.

        See: ``Garden.expanded_metadata`` method
        """
        data = self.expanded_metadata()
        return json.dumps(data, default=garden_json_encoder)

    def validate(self):
        """Helper: perform validation on all fields, even fields which are still defaults."""
        # This is useful for catching fields with a default value which would not
        # otherwise be valid, (e.g.  `None` for any required field), in particular
        # as a sanity check before committing to publishing/registering the user's
        # Garden.

        # This is mostly redundant for any already-set fields, as the validators
        # for those would have (hopefully) already run when they were set. However,
        # (because this is still python), it's relatively easy to *modify* some
        # fields without ever *assigning* to them, which wouldn't trigger
        # validation.

        # Examples
        # --------
        # gc = GardenClient()
        # pea_garden = gc.create_garden(
        #     authors=["Mendel, Gregor"], title="Experiments on Plant Hybridization"
        # )

        # # NOTE: pydantic won't see this, as it's neither construction nor assignment
        # pea_garden.authors.append(None)         # no complaints

        # # checks all fields, even those smuggled in without triggering validation.
        # pea_garden.validate()
        # # ...
        # # ValidationError: 1 validation error for Garden
        # # authors -> 1
        # #   none is not an allowed value (type=type_error.none.not_allowed)
        try:
            _ = self.__class__(**self.dict())
        except ValidationError as err:
            logger.error(err)
            raise

    def _sync_author_metadata(self):
        """helper: authors and contributors of steps and Pipelines also appear as contributors in their respective Pipeline and Garden's metadata."""
        known_contributors = set(self.contributors)
        # garden contributors don't need to duplicate garden authors unless they've been explicitly added
        known_authors = set(self.authors) - known_contributors

        for pipeline in self.pipelines:
            new_contributors = set(pipeline.authors) | set(pipeline.contributors)
            known_contributors |= new_contributors - known_authors

        self.contributors = list(known_contributors)
        return

    def __getattr__(self, name):
        #  note: this is only called as a fallback when __getattribute__ raises an exception,
        #  existing attributes are not affected by overriding this
        message_extra = ""
        for pipeline in self.pipelines:
            short_name = pipeline.short_name
            alias = self.pipeline_aliases.get(short_name) or short_name
            if name == alias:
                return pipeline
            elif name == short_name:
                message_extra = f" Did you mean {alias}?"

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'."
            + message_extra
        )

    def __dir__(self):
        # this gets us jupyter/ipython/repl tab-completion of pipeline names
        return list(super().__dir__()) + self.pipeline_names

    def rename_pipeline(self, old_name: str, new_name: str):
        """Rename a pipeline in this garden.

        Parameters:
            old_name (str): the current short_name of the pipeline
            new_name (str): the new short_name of the pipeline
        Raises:
            ValueError: if the new_name is already in use, or if the old_name is \
            not found, or if the new_name is not a valid identifier.
        """
        if hasattr(self, new_name):
            raise ValueError(
                f"Error: found existing {new_name} attribute on this garden."
            )
        if not hasattr(self, old_name):
            raise ValueError(
                f"Error: could not find pipeline {old_name} on this garden."
            )
        if not new_name.isidentifier():
            raise ValueError("new_name must be a valid identifier.")
        self.pipeline_aliases[old_name] = new_name
        return

    class Config:
        # Configure pydantic per-model settings.

        # We disable validate_all so that pydantic ignores temporarily-illegal defaults
        # We enable validate_assignment to make validation occur naturally even after object construction

        validate_all = False  # (this is the default)
        validate_assignment = True  # validators invoked on assignment
        underscore_attrs_are_private = True


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

        pipelines (list[RegisteredPipeline]):
            List of the pipelines associated with this garden \
            Note that these pipelines can also be accessed directly by their \
            `short_name` (or alias) - see also `pipeline_names` attribute.

        pipeline_names (list[str]):
            List of python identifiers (i.e. variable names) usable for this \
            garden's pipelines. Takes aliases into account.

        pipeline_ids: List[str] = Field(default_factory=list)
        pipeline_aliases: Dict[str, str] = Field(default_factory=dict)

        doi (str):
            A garden's doi usable for citations.

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

    title: str = Field(...)
    authors: List[str] = Field(...)
    contributors: List[str] = Field(default_factory=list, unique_items=True)
    doi: str = Field(...)
    description: Optional[str] = Field(None)
    publisher: str = "Garden-AI"
    year: str = Field(default_factory=lambda: str(datetime.now().year))
    language: str = "en"
    tags: List[str] = Field(default_factory=list, unique_items=True)
    version: str = "0.0.1"
    pipelines: List[RegisteredPipeline] = Field(...)
    pipeline_ids: List[str] = Field(...)
    pipeline_aliases: Dict[str, str] = Field(default_factory=dict)
    # unsure how validation will behave here. the desired behavior is to have a required field that is filled in automagically post_init_post_parse
    pipeline_names: List[str] = Field(init=False)
    _env_vars: Dict[str, str] = PrivateAttr(default_factory=dict)

    @root_validator
    def verify_integrity(cls, values):
        pipelines, pipeline_ids = (
            values.get(attr) for attr in ("pipelines", "pipeline_ids")
        )

        if not len(pipelines) == len(pipeline_ids) and all(
            pipeline.doi in pipeline_ids for pipeline in pipelines
        ):
            raise ValueError(
                "This garden has an invalid state. "
                "Please verify consistency between the `pipelines` and `pipeline_ids` attributes."
            )
        return values

    @validator("year")
    def valid_year(cls, year):
        if len(str(year)) != 4:
            raise ValueError("year must be formatted `YYYY`")
        return str(year)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pipeline_names = [
            self.pipeline_aliases.get(pipeline.short_name) or pipeline.short_name
            for pipeline in self.pipelines
        ]
        return

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
        record = garden.expanded_json()  # this method or plain .json()?
        data = json.loads(record)
        return cls(**data)

    def _repr_html_(self) -> str:
        return garden_repr_html(self)

    def expanded_metadata(self) -> Dict[str, Any]:
        """
        Helper method: builds the "complete" metadata dictionary with nested `Pipeline` and `step` metadata.

        When serializing normally with `garden.{dict(), json()}`, only the DOIs of the pipelines in the garden are included.

        This method returns a superset of `garden.dict()`, so that the following holds:
            valid_garden == Garden(**valid_garden.expanded_metadata()) == Garden(**valid_garden.dict())

        Returns:
            A dictionary containing the `Garden` metadata augmented with a list of `RegisteredPipeline` metadata.

        Raises:
            PipelineNotFoundException: If `garden.pipeline_ids` references an unknown pipeline ID.
        """
        data = self.dict()
        data["pipelines"] = [p.expanded_metadata() for p in self.pipelines]
        return data

    def expanded_json(self) -> JSON:
        """Helper: return the expanded garden metadata as JSON.

        See: ``Garden.expanded_metadata`` method
        """
        data = self.expanded_metadata()
        return json.dumps(data, default=garden_json_encoder)

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
                for doi in self.pipeline_ids
            ],
            version=self.version,
            descriptions=[
                Description(description=self.description, descriptionType="Other")
            ]
            if self.description
            else None,
        ).json()

    def validate(self):
        """Helper: perform validation on all fields, even fields which are still defaults."""
        # This is useful for catching fields with a default value which would not
        # otherwise be valid, (e.g.  `None` for any required field), in particular
        # as a sanity check before committing to publishing/registering the user's
        # Garden.

        # This is mostly redundant for any already-set fields, as the validators
        # for those would have (hopefully) already run when they were set. However,
        # (because this is still python), it's relatively easy to *modify* some
        # fields without ever *assigning* to them, which wouldn't trigger
        # validation.

        # Examples
        # --------
        # gc = GardenClient()
        # pea_garden = gc.create_garden(
        #     authors=["Mendel, Gregor"], title="Experiments on Plant Hybridization"
        # )

        # # NOTE: pydantic won't see this, as it's neither construction nor assignment
        # pea_garden.authors.append(None)         # no complaints

        # # checks all fields, even those smuggled in without triggering validation.
        # pea_garden.validate()
        # # ...
        # # ValidationError: 1 validation error for Garden
        # # authors -> 1
        # #   none is not an allowed value (type=type_error.none.not_allowed)
        try:
            _ = self.__class__(**self.dict())
        except ValidationError as err:
            logger.error(err)
            raise

    def __getattr__(self, name):
        #  note: this is only called as a fallback when __getattribute__ raises an exception,
        #  existing attributes are not affected by overriding this
        message_extra = ""
        for pipeline in self.pipelines:
            short_name = pipeline.short_name
            alias = self.pipeline_aliases.get(short_name) or short_name
            if name == alias:
                return pipeline
            elif name == short_name:
                message_extra = f" Did you mean {alias}?"

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'."
            + message_extra
        )

    def __dir__(self):
        # this gets us jupyter/ipython/repl tab-completion of pipeline names
        return list(super().__dir__()) + self.pipeline_names

    class Config:
        # Configure pydantic per-model settings.

        # We disable validate_all so that pydantic ignores temporarily-illegal defaults
        # We enable validate_assignment to make validation occur naturally even after object construction

        validate_all = False  # (this is the default)
        validate_assignment = True  # validators invoked on assignment
        underscore_attrs_are_private = True


def garden_repr_html(garden: Union[Garden, PublishedGarden]) -> str:
    style = "<style>th {text-align: left;}</style>"
    title = f"<h2>{garden.title}</h2>"
    details = f"<p>Authors: {', '.join(garden.authors)}<br>DOI: {garden.doi}</p>"
    pipelines = "<h3>Pipelines</h3>" + tabulate(
        [
            {
                key.title(): str(getattr(pipeline, key))
                for key in ("title", "authors", "doi")
            }
            for pipeline in garden.pipelines
        ],
        headers="keys",
        tablefmt="html",
    )
    optional = "<h3>Additional data</h3>" + tabulate(
        [
            (field, val)
            for field, val in garden.dict().items()
            if field not in ("title", "authors", "doi")
            and "pipeline" not in field
            and val
        ],
        tablefmt="html",
    )
    return style + title + details + pipelines + optional
