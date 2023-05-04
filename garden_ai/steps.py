from __future__ import annotations

import json
import logging
import typing
from functools import update_wrapper, wraps
from inspect import Parameter, Signature, signature
from typing import Any, Callable, Dict, List, Optional

from pydantic import Field, validator
from pydantic.dataclasses import dataclass
from typing_extensions import get_type_hints

from garden_ai.mlmodel import Model, _Model
from garden_ai.utils.misc import JSON, garden_json_encoder

logger = logging.getLogger()


class DataclassConfig:
    # pydantic dataclasses read their config via decorator argument, not as
    # nested class (like BaseModels do)
    validate_assignment = True


@dataclass(config=DataclassConfig)
class Step:
    """The ``Step`` class (via the ``@step`` decorator) wraps a callable for use as a single step in a ``Pipeline``.

    **IMPORTANT**: When included in a ``Pipeline``, all ``Step``s will be checked
    for composability with respect to their argument- and return-type
    annotations.  This requires that the decorated function has complete
    argument- and return-type annotations; a ``Step`` will fail to initialize with
    an exception if this condition is not met. Note that there is (currently) no
    type-checking at runtime; annotations are taken in good faith.

    See "Notes" below for additional discussion on composing steps with multiple
    arguments by returning a properly-typed ``tuple`` as a result.

    Attributes
    ----------
    func: Callable
        ``func`` is whatever should be called when this ``Step`` is reached in its
        ``Pipeline``, and is passed the output of the previous step (if one
        exists) as argument(s).  Typically a plain python function, but a
        callable object with a sufficiently-annotated ``__call__`` magic method is
        also acceptable. When the ``@step`` decorator is used, ``func`` will be the
        decorated function. To be composable as a Step, the following are
        *required*:
            1. For all but the first Step in a pipeline, ``func`` must only
               require positional arguments, each of which must be annotated
            2. For any but the last Step in a pipeline, ``func`` must return
               either a single object (if subsequent Step has a single
               positional arg) or a tuple (if subsequent Step has multiple
               positional args)

    title: str
        An official name or title for the Step. Currently, ``func.__name__`` is
        used as a default.

    description: str
        A human-readable description of the step. Currently, ``func.__doc__`` is
        used as a default.

    input_info: str
        Human-readable description of the input data and/or relevant
        characteristics that *should* hold true for this step's input (e.g.
        dimensions of a matrix or column names of a dataframe). Currently,
        ``typing.get_type_hints(func, include_extras=True)`` is used for the
        default.

    output_info: str
        Human-readable description of the output data and/or relevant
        characteristics that *will* hold true for this step's output (e.g.
        dimensions of a matrix or column names of a dataframe). Currently,
        ``typing.get_type_hints(func, include_extras=True)`` is used for the
        default.


    authors: List[str]
        The main researchers involved in producing the Step, for citation and discoverability
        purposes.

    model_uris: List[str]
        A reference to the models used in this step, if any.
        Model identifiers as stored in MLFlow (not including the 'models:/' prefix).

    uuid: UUID
        short for "uuid"

    Raises
    ------
    TypeError
        If ``func`` has any arguments without annotations, is missing a return
        annotation, or uses ``None`` or ``Any`` as annotation.

    Notes
    -----
    We require annotations because we need ``Step``s to be composable functions
    in their respective ``Pipeline``s. However, due to python's highly flexible
    ``*args`` syntax, function composition is inherently ambiguous -- e.g. if we
    wish to compose ``f`` with ``g`,` and ``g` returns a tuple of values, should
    that tuple be passed to ``f`` as a tuple (i.e. ``f(g(*args))``), or should
    it be unpacked as individual arguments to ``f`` (i.e. ``f(*g(*args))``,
    noting the extra ``*``)?

    To resolve this ambiguity, we could either (a) restrict the set of acceptable
    callables to be only those with a single argument and a single return value, or
    (b) rely on thorough function annotations to disambiguate. While best practices
    are likely to stick to single-input-single-output ``Step``s, we currently try
    to support (b) by composing steps together differently if ``g` seems to be
    returning an "argument tuple" for ``f`` as follows:
        - ``g` has a return annotation indicating the types within the tuple:
            e.g. ``def g(...) -> tuple[str, int, pd.DataFrame]``
        - ``f`` has exactly those argument annotations, in the same order:
            e.g. ``def f(x: str, y: int, z: pd.DataFrame) -> ...``
        - if both are true, compose ``f`` with ``g`` by **unpacking the tuple when it returns**:
            e.g. ``f(*g(*args))`` (noting the extra ``*``), otherwise ``f(g(*args))`` - like any other Step.
    """

    func: Callable
    authors: List[str] = Field(default_factory=list)
    contributors: List[str] = Field(default_factory=list)
    title: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    input_info: Optional[str] = Field(None)
    output_info: Optional[str] = Field(None)
    conda_dependencies: List[str] = Field(default_factory=list)
    pip_dependencies: List[str] = Field(default_factory=list)
    python_version: Optional[str] = Field(None)
    model_uris: List[str] = Field(default_factory=list)

    def __post_init_post_parse__(self):
        # like __post_init__, but called after pydantic validation
        # copies e.g. __doc__ and __name__ from
        # the underlying callable to this object
        # (also handy for signature/annotations)
        update_wrapper(self, self.func)
        self.title = self.title or self.__name__
        self.description = self.description or self.__doc__
        self.__signature__ = signature(self.func)
        input_hints: Dict = get_type_hints(self.func, include_extras=True)
        return_hint = input_hints.pop("return")
        if self.input_info is None:
            self.input_info = str(input_hints)
        if self.output_info is None:
            self.output_info = f"return: {return_hint}"
        self._infer_model_deps()
        return

    def __call__(self, *args, **kwargs):
        # keep it simple: just pass input the underlying callable.
        return self.func(*args, **kwargs)

    def _infer_model_deps(self):
        """
        If this step's function has a Model as a default argument, like
        ``func(*args, model=Model(...))``, extract the dependencies for that model
        and track them as step-level dependencies.
        """

        sig = signature(self.func)
        for param in sig.parameters.values():
            if isinstance(param.default, _Model):
                model = param.default
                self.python_version = model.python_version
                self.conda_dependencies += model.conda_dependencies
                self.pip_dependencies += model.pip_dependencies
                self.model_uris += [model.model_full_name]
        return

    @validator("func")
    def has_annotations(cls, f: Callable):
        sig = signature(f)
        # check that any positional arguments have annotations
        for p in sig.parameters.values():
            if p.annotation is Parameter.empty is p.default:
                # fine to skip annotation only if there's a default we can use to infer type
                raise TypeError(
                    f"Parameter {p} is missing an annotation in {f.__name__}'s definition. "
                    "Please double check that the argument list is fully annotated.\n"
                    "See also: https://peps.python.org/pep-0484/#type-definition-syntax"
                )
        # check that return value has annotation
        if sig.return_annotation in {Signature.empty, None}:
            raise TypeError(
                f"{f.__name__}'s definition is missing a return annotation, or returns None.\n"
                "See also: https://peps.python.org/pep-0484/#type-definition-syntax"
            )
        return f

    @validator("func")
    def uses_Any(cls, f: Callable):
        sig = signature(f)
        # check input args
        for p in sig.parameters.values():
            if p.annotation is typing.Any:
                raise TypeError(
                    f"In {f.__name__}'s definition, found `Any` annotating the parameter {p}, \n "
                    "which would prevent us from verifying that steps would compose together correctly\n"
                    "before being published as a Pipeline. Please try again with a more descriptive type hint.\n"
                    "We use `beartype` to resolve type hints -- for a full list of supported annotations \n "
                    "(including 3rd party type hints, like `numpy.typing`), see:\n "
                    "https://github.com/beartype/beartype#compliance"
                )
        if sig.return_annotation is typing.Any:
            raise TypeError(
                f"In {f.__name__}'s definition, found `Any` as the return annotation, \n "
                "which would prevent us from verifying that steps would compose together correctly\n"
                "before being published as a Pipeline. Please try again with a more descriptive type hint.\n"
                "We use `beartype` to resolve type hints -- for a full list of supported annotations \n "
                "(including 3rd party type hints, like `numpy.typing`), see:\n "
                "https://github.com/beartype/beartype#compliance"
            )
        return f

    def json(self) -> JSON:
        return json.dumps(self, default=garden_json_encoder)

    def dict(self) -> Dict[str, Any]:
        d = {}
        for key in self.__dataclass_fields__:
            val = getattr(self, key)
            d[key] = val
        return d


def step(func: Callable = None, **kwargs):
    """Helper: provide ``@step(...)`` decorator for creation of ``Step``s."""
    # note:
    # The ``Step`` class itself could also technically be used as a decorator,
    # but would run into trouble as soon as you tried passing any arguments:
    # this definition means ``@step`` and ``@step(...)`` are equivalent decorators,
    # because ``@Step`` and ``@Step(...)`` could not be.
    if func is not None:
        # called like ``@step``
        # (or ``my_func = step(my_func, **kwargs)``)
        data = {**kwargs, "func": func}
        return Step(**data)  # type: ignore

    else:
        # called like ``@step(**kwargs)``
        def wrapper(f):
            data = {**kwargs, "func": f}
            return Step(**data)

        return wrapper


def inference_step(model_uri: str, **kwargs):
    """Helper: provide ``@inference_step(...)`` decorator for creation of ``Step``s.

    Example:
    --------
        ```python
        @inference_step(model_uri="me@uni.edu-my-model/version")
        def my_step(data: pd.DataFrame) -> MyResultType:
            pass  # NOTE: leave the function body empty

        ## equivalent to:
        @step
        def my_step(
            data: MyDataType,
            model=garden_ai.Model("me@uni.edu-my-model/version"),
        ) -> MyResultType:
            return model.predict(data)
        ```
    """

    def wrapper(f: Callable):
        @wraps(f)  # make sure we aren't losing signature info
        def boilerplate(*args, model=Model(model_uri), **_kwargs):
            return model.predict(*args)

        return step(boilerplate)

    return wrapper
