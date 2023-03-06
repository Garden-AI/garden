import logging
import sys
from inspect import Signature, signature, Parameter
from itertools import zip_longest
import json
import base64

import beartype.door
import requests
from typing_extensions import TypeAlias

if sys.version_info < (3, 9):
    from typing_extensions import get_args, get_origin
else:
    from beartype.typing import get_args, get_origin

from pydantic.json import pydantic_encoder

JSON: TypeAlias = str

logger = logging.getLogger()
issubtype = beartype.door.is_subhint


# for contrast: unsafe_compose = lambda f,g: lambda *args,**kwargs: f(g(*args, **kwargs))
def safe_compose(f, g):
    """Helper: compose function `f` with function `g`, provided their annotations indicate compatibility.

    Arguments with defaults are ignored.

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

    f_in = tuple(
        p.annotation for p in f_sig.parameters.values() if p.default is Parameter.empty
    )
    g_out = g_sig.return_annotation

    if get_origin(g_out) is tuple and len(f_in) > 1:
        # case 1: g returns a tuple which, *if unpacked*, may align with f's annotations
        # for example:
        # g defined like `def g(...) -> tuple[A, B, C]: ...`, and
        # f defined like `def f(a: A, b: B, c: C) -> ...`
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


def requests_to_curl(response: requests.Response) -> str:
    """Given a `Response` object, build a cURL command equivalent to the request which prompted it.

    Useful for debugging with e.g. Postman (or cURL, of course).

    Example:
    --------
    res = requests.post(...)
    print(requests_to_curl(res))
    """
    request = response.request
    method = request.method
    uri = request.url
    headers = " -H ".join(f'"{k}: {v}"' for k, v in request.headers.items())
    if request.body is None:
        return f"curl -X {method} -H {headers} {uri}"
    if isinstance(request.body, bytes):
        data: str = request.body.decode()
    else:
        data = request.body
    return f"curl -X {method} -H {headers} -d '{data}' {uri}"


def extract_email_from_globus_jwt(jwt: str) -> str:
    try:
        # jwts are three base64 encoded segments delimited by periods.
        _, payload_b64, _ = jwt.split(".")
        payload = json.loads(base64.b64decode(payload_b64 + "===").decode("utf-8"))
    except Exception as e:
        raise Exception("Invalid JWT") from e
    try:
        email = payload["identity_set"][0]["email"]
    except KeyError as e:
        raise Exception("JWT did not include user email") from e
    return email
