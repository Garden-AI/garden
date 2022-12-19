from inspect import Signature, signature
from itertools import zip_longest
from typing import Union

from typing_extensions import get_args, get_origin

try:  # noqa C901 (mccabe exceeded due to try/except)
    from types import UnionType  # type: ignore
except ImportError:

    def issubtype(a: type, b: type) -> bool:
        """Helper: report whether `a` is a subtype of `b`.

        This is defined slightly differently for versions which do not know about
        `types.UnionType` (the `type(A | B)` special type)
        """
        if a == b:
            return True
        if a is None or b is None:
            return False
        # e.g. if a is `Union[x,y]`, then a_origin is `Union` and a_args is `(x, y)`
        a_origin, a_args = get_origin(a), get_args(a)
        b_origin, b_args = get_origin(b), get_args(b)
        if a_args == b_args == ():
            # case 1: no args
            # (i.e. rules out anything fancy enough for square brackets)
            return issubclass(a, b)

        elif b_origin is Union:
            # case 2: b is a union of types, which are the contents of b_args
            if a_origin is Union:
                # both are unions, so must have *all* the same args
                return all(issubtype(x, y) for (x, y) in zip_longest(a_args, b_args))
            elif a_origin is None:
                # builtin `issubclass` treats union types how you'd expect for
                # literally this case only, might as well use it
                return issubclass(a, b)
            else:
                # a is a complex but non-union type, so *must* subtype one of b_args
                return any(issubtype(a, t) for t in b_args)

        elif issubclass(a_origin or a, b_origin or b):
            # Case 3: other complex types, e.g. List[str] subtypes Sequence[str]
            # the little `or`s are here because of the following fun fact:
            # get_origin(tuple[x]) == get_origin(Tuple[x]) == get_origin(Tuple) == tuple
            # ... get_origin(tuple) is None.
            # I'm sure there's a very wise reason for this
            return all(issubtype(x, y) for x, y in zip(a_args, b_args))

        return False

else:

    def issubtype(a: type, b: type) -> bool:
        """Helper: report whether `a` is a subtype of `b`.

        This works for many cases, e.g. `issubtype(X, X | Y) == True` or
        `issubtype(Union[X,Y], Union[X,Y,Z])`, which are inconsistent and/or
        exceptions in the analogous `issubclass` or `isinstance` builtins.

        """
        if a == b:
            return True
        if a is None or b is None:
            return False
        # e.g. if a is `Union[x,y]`, then a_origin is `Union` and a_args is `(x, y)`
        a_origin, a_args = get_origin(a), get_args(a)
        b_origin, b_args = get_origin(b), get_args(b)
        if a_args == b_args == ():
            # case 1: no args; simple types
            return issubclass(a, b)

        elif b_origin is Union or isinstance(b, UnionType):
            # case 2: b is a union of some types, possibly including a
            # (either Union[a, _] or (a | _) -style)
            # note that typing.Union cannot be used with isinstance(), but
            # types.UnionType can be.
            if a_origin is Union or isinstance(a, UnionType):
                # both are unions, so must have *all* the same args
                return all(issubtype(x, y) for (x, y) in zip_longest(a_args, b_args))
            elif a_origin is None:
                # a is a simple type, so `issubclass(a, Union[a, ...])` will report True
                # `issubclass` handles union types how you'd expect for
                # literally this case only, might as well use it
                return issubclass(a, b)
            else:
                # a is a complex but non-union type, so *must* subtype one of b_args
                return any(issubtype(a, t) for t in b_args)

        elif issubclass(a_origin or a, b_origin or b):
            # Case 3: other complex types, e.g. List[str] subtypes Sequence[str]
            # the little `or`s are here because of the following fun fact:
            # get_origin(tuple[x]) == get_origin(Tuple[x]) == get_origin(Tuple) == tuple
            # ... get_origin(tuple) is None.
            # I'm sure there's a very wise reason for this
            return all(issubtype(x, y) for x, y in zip(a_args, b_args))

        return False


# for contrast: unsafe_compose = lambda f,g: lambda *args,**kwargs: f(g(*args, **kwargs))
def safe_compose(f, g):
    """Helper: compose function `f` with function `g`, provided their annotations indicate compatibility.

    This is smart enough to figure out whether `g`'s result is meant as an
    `*args` tuple for `f`, or if it's meant as a plain return value (which might
    still be a tuple). Complains with an exception if the signatures don't match.


    Parameters
    ----------
    f : Callable
        `f` is any callable which:
            1. Has complete argument and return type annotations (steps are validated for this).
            2. accepts 1 or more positional arguments, corresponding to `g`'s return type.
            3. If `g` returns a tuple, `f` should either accept the
                unpacked *elements* of the tuple as a list of arguments;
                else `f` should accept a tuple itself.

    g : Callable
        like `f`, `g` can be any callable which:
            1. Has complete argument and return type annotations.
            2. When `f` expects a single argument, returns a single python
                object of the appropriate type.
            3. When `f` expects multiple arguments, returns a tuple with
                appropriately-typed elements.

    Raises
    ------
    TypeError
        If the annotations for `f`'s argument types and `g`'s return
        type are not equivalent.
    """

    f_sig: Signature = signature(f)
    g_sig: Signature = signature(g)

    f_in = tuple(p.annotation for p in f_sig.parameters.values())
    g_out = g_sig.return_annotation

    if get_origin(g_out) is tuple and len(f_in) > 1:
        # case 1: g returns a tuple which, *if unpacked*, may align with f's annotations
        # for example:
        # g defined like `def g(...) -> tuple[A, B, C]: ...`, and
        # f defined like `def f(a: A, b: B, c:C) -> ...`
        g_returns: tuple = get_args(g_out)
        if all(
            issubtype(output_type, input_type)
            for (output_type, input_type) in zip_longest(g_returns, f_in)
        ):
            # note that we unpack g's output
            def f_of_g(*args, **kwargs):
                return f(*g(*args, **kwargs))

        else:
            raise TypeError(
                (
                    f"Could not compose step {f.__name__} with step {g.__name__} "
                    "due to return type signature mismatch: "
                    f"expected tuple[{f_in}], got {g_out}."
                )
            )
    elif len(f_in) == 1:
        # case 2: return is a single value; verify that it's the only one
        # expected by f.
        if issubtype(g_out, f_in[0]):

            # note that we do NOT unpack g's output
            def f_of_g(*args, **kwargs):
                return f(g(*args, **kwargs))

        else:
            raise TypeError(
                (
                    f"Could not compose step {f.__name__} with step {g.__name__} "
                    "due to return type signature mismatch: "
                    f"expected {f_in[0]}, got {g_out}."
                )
            )
    else:
        # case 3: signatures are neither single types nor equivalent tuples
        raise TypeError(
            (
                f"Could not compose with step {g.__name__} due to return "
                "type signature mismatch. Please double-check that its "
                "return signature matches the argument(s) of the "
                "subsequent step."
            )
        )

    # give the returned function a new signature, corresponding
    # to g's input types and f's return type
    f_of_g.__signature__ = Signature(
        parameters=g_sig.parameters.values(),
        return_annotation=f_sig.return_annotation,
    )
    f_of_g.__name__ = f.__name__ + "_COMPOSED_WITH_" + g.__name__
    return f_of_g
