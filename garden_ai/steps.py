from __future__ import annotations

import inspect
import json
import logging
import typing
from functools import update_wrapper
from inspect import Parameter, Signature, signature
from typing import Any, Callable, Dict, List, Optional

from pydantic import Field, validator
from pydantic.dataclasses import dataclass
from typing_extensions import get_type_hints

from garden_ai.utils.misc import JSON, garden_json_encoder

logger = logging.getLogger()


class DataclassConfig:
    # pydantic dataclasses read their config via decorator argument, not as
    # nested class (like BaseModels do)
    validate_assignment = True


@dataclass(config=DataclassConfig)
class Step:
    """The ``Step`` class (via the ``@step`` decorator) wraps a callable for use as a single step in a ``Pipeline``.

    IMPORTANT:
        When included in a ``Pipeline``, all steps will be checked for \
        composability with respect to their argument- and return-type \
        annotations.  This requires that the decorated function has complete \
        argument- and return-type annotations; a ``Step`` will fail to \
        initialize with an exception if this condition is not met. Note that \
        there is (currently) no type-checking at runtime. \
        See "Notes" below for additional discussion on composing steps which \
        may take multiple arguments by returning a properly-typed `tuple` as a result.

    Attributes:
        func (Callable):
            The well-annotated function that should be called when this ``Step`` \
            is reached in its ``Pipeline``, which is passed the output of the \
            previous step (if one exists) as input. This is the only required \
            attribute, all others are optional or should not be modified.
        title (str):
            An official name or title for the Step. Currently, ``func.__name__`` is used as a default.
        description (str):
            A human-readable description of the step. Currently, ``func.__doc__`` is used as a default.
        input_info (str):
            Human-readable description of the input data and/or relevant
            characteristics that *should* hold true for this step's input (e.g.
            dimensions of a matrix or column names of a dataframe).
        output_info (str):
            Human-readable description of the output data and/or relevant
            characteristics that *will* hold true for this step's output (e.g.
            dimensions of a matrix or column names of a dataframe).
        authors (List[str]):
            The main researchers involved in producing the Step, for citation and discoverability purposes.
        model_full_names (List[str]):
            A reference to the models used in this step, if any. Model identifiers are as stored in MLFlow (not including the 'models:/' prefix).
        source (Optional[str]):
            Should not be set by users. Consists of the plain python source code \
            used to define `func`, if possible.

    Raises:
        TypeError:
            If ``func`` has any arguments without annotations, is missing a return annotation, or uses ``None`` or ``Any`` as annotation.

    Notes:
        Due to python's flexible `*args` syntax, function composition can be \
        ambiguous -- e.g. if we wish to compose `f` with `g`, and `g` returns a \
        tuple of values, should that tuple be passed to `f` as a tuple (i.e. \
        ``f(g(*args))``), or should it be unpacked as individual arguments to \
        `f` (i.e. `f(*g(*args))`, noting the extra `*`)?

        To resolve this ambiguity, we could either (a) restrict the set of \
        acceptable  callables to be only those with a single argument and a \
        single return value, or  (b) rely on function annotations to \
        disambiguate. While best practices  are likely to stick to simpler, \
        single-input-single-output `Step`s, we currently try  to support (b) by \
        composing steps together differently if `g` seems to be returning an \
        "argument tuple" for `f`. For example, if `g` returns `... -> tuple[T1, \
        T2, T3]` and `f` is annotated `def f(a: T1, b: T2, c: T3)`, we unpack \
        `g`'s output before passing it to `f`.
    """

    func: Callable
    authors: List[str] = Field(default_factory=list)
    contributors: List[str] = Field(default_factory=list)
    title: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    input_info: Optional[str] = Field(None)
    output_info: Optional[str] = Field(None)
    model_full_names: List[str] = Field(default_factory=list)
    source: Optional[str] = Field(None)

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
        self._track_models()
        return

    def __call__(self, *args, **kwargs):
        # keep it simple: just pass input the underlying callable.
        return self.func(*args, **kwargs)

    def _track_models(self):
        """
        If this step's function has a Model as a default argument, like
        ``func(*args, model=Model(...))``, record the model name
        """

        # if `_Model` was never defined in main, there must not be any
        try:
            from __main__ import _Model
        except ImportError:
            return

        sig = signature(self.func)
        for param in sig.parameters.values():
            if isinstance(param.default, _Model):
                model = param.default
                self.model_full_names += [model.full_name]
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

    @validator("source", always=True, pre=False)
    def has_findable_source(cls, _, values):
        # ignores any prior value for the "source" field, populating it with the
        # found source of `func`.  There are tons of edge cases if the user is
        # passing an arbitrary callable directly to the Step constructor, but
        # because only plain python can be decorated, if they're using the
        # decorator this shouldn't be problematic.
        if "func" in values:
            func = values["func"]
            try:
                return inspect.getsource(func)
            except (OSError, TypeError) as e:
                raise ValueError(
                    f"Could not find python source code for {func}. If using a \
builtin or externally-defined function as a step, best practice is \
to use @step to decorate a minimal function that invokes it, rather \
than using it as a step directly."
                ) from e

    def json(self) -> JSON:
        return json.dumps(self, default=garden_json_encoder)

    def dict(self) -> Dict[str, Any]:
        d = {}
        for key in self.__dataclass_fields__:
            val = getattr(self, key)
            d[key] = val
        return d


def step(func: Callable = None, **kwargs):
    """Helper: provide `@step` /`@step(...)` decorator for creation of `Steps`."""
    # note:
    # The capital-S `Step class itself could also technically be used as a decorator,
    # but would run into trouble as soon as you tried passing any arguments:
    # this definition means `@step` and `@step(...)` are equivalent decorators,
    # because `@Step` and `@Step(...)` could not be.
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
