from __future__ import annotations

import logging
from functools import update_wrapper
from inspect import Parameter, Signature, signature
from typing import Callable, List, Optional
from uuid import UUID, uuid4

from pydantic import Field, validator
from pydantic.dataclasses import dataclass


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
