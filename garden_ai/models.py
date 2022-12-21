from __future__ import annotations

import logging
from datetime import datetime
from functools import reduce, update_wrapper
from inspect import Parameter, Signature, signature
from typing import Any, Callable, List, Optional, Tuple, cast
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ValidationError, validator
from pydantic.dataclasses import dataclass
from pydantic.json import pydantic_encoder

from garden_ai.utils import safe_compose

logger = logging.getLogger()


class DataclassConfig:
    # pydantic dataclasses read their config via decorator argument, not as
    # nested class (like BaseModels do)
    validate_assignment = True
    underscore_attrs_are_private = True


@dataclass(config=DataclassConfig)
class Step:
    """The `Step` class (via the `@step` decorator) wraps a callable for use as a single step in a `Pipeline`.

    **IMPORTANT**: When included in a `Pipeline`, all `Step`s will be checked
    for composability with respect to their argument- and return-type
    annotations.  This requires that the decorated function has complete
    argument- and return-type annotations; a `Step` will fail to initialize with
    an exception if this condition is not met. Note that there is (currently) no
    type-checking at runtime; annotations are taken in good faith.

    See "Notes" below for additional discussion on composing steps with multiple
    arguments.

    Attributes
    ----------
    func: Callable
        `func` is whatever should be called when this `Step` is reached in its
        `Pipeline`, and is passed the output of the previous step (if one
        exists) as argument(s).  Typically a plain python function, but a
        callable object with a sufficiently-annotated `__call__` magic method is
        also acceptable. When the `@step` decorator is used, `func` will be the
        decorated function. To be composable as a Step, the following are
        *required*:
            1. For all but the first Step in a pipeline, `func` must only
               require positional arguments, each of which must be annotated
            2. For any but the last Step in a pipeline, `func` must return
               either a single object (if subsequent Step has a single
               positional arg) or a tuple (if subsequent Step has multiple
               positional args)
        This ought to be a `pure function<https://en.wikipedia.org/wiki/Pure_function>`_, or as close to one as possible.

    title: str
        An official name or title for the Step. Currently, `func.__name__` is
        used as a default.

    description: str
        A human-readable description of the step. Currently, `func.__doc__` is
        used as a default.

    input_info: str
        Human-readable description of the input data and/or relevant
        characteristics that *should* hold true for this step's input (e.g.
        dimensions of a matrix or column names of a dataframe).

    output_info: str
        Human-readable description of the output data and/or relevant
        characteristics that *will* hold true for this step's output (e.g.
        dimensions of a matrix or column names of a dataframe).


    authors: List[str]
        The main researchers involved in producing the Step, for citation and discoverability
        purposes. Behavior of this is currently TBD:
            - Do we want authorship to propagate to/from steps to pipelines/gardens?
            - Do we want authorship to propagate as a "contributor"? How far should it go?
            - Do steps need contributors?
    uuid: UUID
        short for "uuid"

    Raises
    ------
    TypeError
        If `func` has any arguments without annotations, is missing a return
        annotation, or has `None` as a return annotation.

    Notes
    -----
    We require annotations because we need `Step`s to be composable functions in
    their respective `Pipeline`s. However, due to python's highly flexible
    `*args` syntax, function composition is inherently ambiguous -- e.g. if we
    wish to compose `f` with `g`, and `g` returns a tuple of values, should that tuple be passed to `f`
    as a tuple (i.e. `f(g(*args))`), or should it be unpacked as individual
    arguments to `f` (i.e. `f(*g(*args))`, noting the extra ``*``)?

    To resolve this ambiguity, we could either (a) restrict the set of acceptable
    callables to be only those with a single argument and a single return value, or
    (b) rely on thorough function annotations to disambiguate. While best practices
    are likely to stick to single-input-single-output `Step`s, we currently try
    to support (b) by composing steps together differently if `g` seems to be
    returning an "argument tuple" for `f` as follows:
        - `g` must have annotations indicating the types within the tuple, e.g. `def g(...) -> tuple[str, int, pd.DataFrame]`
        - `f` must have exactly those argument annotations, e.g. `def f(x: str, y: int, z: pd.DataFrame) -> ...`
        - if both are true, compose by unpacking like `f(*g(*args))`, otherwise `f(g(*args))` like any other Step.
    """

    func: Callable
    authors: List[str] = Field(default_factory=list)
    contributors: List[str] = Field(default_factory=list)
    title: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    input_info: Optional[str] = Field(None)
    output_info: Optional[str] = Field(None)
    uuid: UUID = Field(default_factory=uuid4)

    def __post_init_post_parse__(self):
        # like __post_init__, but called after pydantic validation
        # copies e.g. __doc__ and __name__ from
        # the underlying callable to this object
        # (also handy for signature/annotations)
        update_wrapper(self, self.func)
        self.title = self.title or self.__name__
        self.description = self.description or self.__doc__
        self.__signature__ = signature(self.func)
        return

    def __call__(self, *args, **kwargs):
        # keep it simple: just pass input the underlying callable.
        # (though if we were going to do it anywhere, here's where we'd add
        # logic to check that the types are what we expected them to be at
        # runtime. see also: pydantic validate_arguments for inspiration)
        return self.func(*args, **kwargs)

    @validator("func")
    def has_annotations(cls, f: Callable):
        sig = signature(f)
        # check that any positional arguments have annotations
        # maybe: warn about kwargs if any?
        for p in sig.parameters.values():
            if p.annotation is Parameter.empty:
                raise TypeError(
                    f"Parameter {p} is missing an annotation in {f.__name__}'s definition. "
                    "Please double check that the argument list is fully annotated.\n"
                    "See also: https://peps.python.org/pep-0484/#type-definition-syntax"
                )
        if sig.return_annotation in {Signature.empty, None}:
            raise TypeError(
                f"{f.__name__}'s definition is missing a return annotation, or returns None.\n"
                "See also: https://peps.python.org/pep-0484/#type-definition-syntax"
            )
        return f

    def register(self):
        """NOT IMPLEMENTED

        This method could be used to "register" the Step with funcx
        """
        raise NotImplementedError


def step(func: Callable = None, **kwargs):
    """Helper: provide decorator interface/syntax sugar for `Step`s."""
    # note:
    # while the Step class itself could also be used as a decorator to create a
    # basically identical object, it's not possible to also pass kwargs (like a
    # title or author) to the Step constructor without something like this.
    if func is not None:
        data = {**kwargs, "func": func}
        return Step(**data)  # type: ignore

    else:

        def wrapper(f):
            data = {**kwargs, "func": f}
            return Step(**data)

        return wrapper


# this may not be useful for most IDEs, but in a repl/notebook, `help(step)`
# should report the more detailed docstring.
# TODO does this mess with sphinx?
step.__doc__ = Step.__doc__


@dataclass(config=DataclassConfig)
class Pipeline:
    """
    The `Pipeline` class represents a sequence of steps
    that form a pipeline. It has a list of authors, a title,
    and a list of steps. The __call__ method can be used
    to execute the pipeline by calling each Step in order
    with the output of the previous Step as the input to the
    next Step. The register method can be used to register
    each Step in the pipeline.

    Args:
    authors (List[str]): A list of the authors of the pipeline.
    title (str): The title of the pipeline.
    steps (List[Step]): A list of the steps in the pipeline.

    """

    title: str = Field(...)
    authors: List[str] = Field(...)
    steps: Tuple[Step, ...] = Field(...)
    contributors: List[str] = Field(default_factory=list, unique_items=True)
    doi: str = cast(str, Field(default_factory=lambda: None))
    uuid: UUID = Field(default_factory=uuid4)
    # note: tuple vs list decision; a list of authors is conceptually more mutable than
    # the list of steps ought to be, but maybe we should just use tuples everywhere?

    def _composed_steps(*args, **kwargs):
        """ "This method intentionally left blank"

        We define this as a stub here, instead setting it as an attribute in
        `__post_init_post_parse__`, which is the earliest point after we
        validate that the steps are composable that we could modify the Pipeline
        object.
        This indirection is only necessary because `__call__` itself is looked
        up at the class level, so can't be set dynamically for different instances.
        """
        raise NotImplementedError

    @validator("steps")
    def check_steps_composable(cls, steps):
        if len(steps) == 0:
            raise ValueError("Cannot have no steps in a pipeline.")
        try:
            reduce(safe_compose, reversed(steps))
        except TypeError as e:
            logger.error(e)
            raise
        return steps

    def register_pipeline(self):
        """register this `Pipeline`'s complete Step composition as a funcx function
        TODO
        """
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._composed_steps(*args, **kwargs)

    def __post_init_post_parse__(self):
        """Build a single composite function from this pipeline's steps.

        I think it seems reasonable for the default to compose each Step
        together like this so that the entire pipeline can be run as the same
        funcx function, but we might want to think about how/why we might let
        users opt out of this behavior in certain cases.
        """
        self._composed_steps = reduce(safe_compose, reversed(self.steps))
        self.__signature__ = signature(self._composed_steps)
        self._sync_author_metadata()
        return

    def _sync_author_metadata(self):
        known_authors = set(self.authors)
        known_contributors = set(self.contributors)
        for step in self.steps:
            new_contributors = set(step.authors) | set(step.contributors)
            known_contributors |= new_contributors - known_authors
        self.contributors = list(known_contributors)
        return

    def register(self):
        raise NotImplementedError


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
    To remedy this, if the doi field is unset when registering the
    garden, we could just build one for the user with the datacite api.
    This could also eventually be exposed as a `register_doi()` method.
    """

    #
    __required__: List[str] = ["authors", "title", "doi"]
    __recommended__: List[str] = ["description", "tags", "version"]

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

    resourceTypeGeneral: str = "Other"  # (or: model, software, service, interactive?)
    publisher: str = "Garden"
    year: str = Field(default_factory=lambda: str(datetime.now().year))
    language: str = "en"
    tags: List[str] = Field(default_factory=list, unique_items=True)
    version: str = "0.0.1"  # TODO: enforce semver for this?

    pipelines: List[Pipeline] = Field(default_factory=list)

    # field(s) for which we might want to ''disable'' mutation
    garden_id: UUID = Field(default_factory=uuid4, allow_mutation=False)

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
        # TODO this should eventually hit the datacite api

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

    def json(self):
        def garden_json_encoder(obj):
            """workaround: pydantic supports custom encoders for all but built-in types.

            In our case, this means we can't specify how to serialize
            `function`s (like in every Step) in pydantic; there is an open PR to
            fix this - https://github.com/pydantic/pydantic/pull/2745 - but it's
            been in limbo for over a year, so this is the least-hacky option in
            the meantime.
            """
            if isinstance(obj, type(lambda: None)):
                # ^b/c isinstance(obj, function) can't work for ~reasons~ 🐍
                return f"{obj.__name__}: {signature(obj)}"
            else:
                return pydantic_encoder(obj)

        return super().json(encoder=garden_json_encoder)

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

    def add_new_pipeline(
        self, title: str, steps: List[Step], authors: List[str] = None, **kwargs
    ):
        """Create a new Pipeline object and add it to this Garden's list of pipelines.
        (arguments have the same meaning as counterparts in `Pipeline` constructor)
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
