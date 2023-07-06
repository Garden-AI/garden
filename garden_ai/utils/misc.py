import base64
import json
import logging
import re
import sys
from inspect import Parameter, Signature, signature
from keyword import iskeyword
from typing import Callable, List, Optional, Tuple

import beartype.door
import requests
import yaml
from packaging.requirements import InvalidRequirement, Requirement
from pydantic.json import pydantic_encoder
from typing_extensions import TypeAlias

JSON: TypeAlias = str

logger = logging.getLogger()
issubtype = beartype.door.is_subhint


def safe_compose(f: Callable, g: Callable):
    """Helper: compose function `f` with function `g`, provided their annotations indicate compatibility.
    Arguments with defaults are ignored.

    Parameters
    ----------
    f : Callable
    g : Callable

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

    if len(f_in) == 1:
        if issubtype(g_out, f_in[0]):
            # note that we do NOT unpack g's output
            def f_of_g(*args, **kwargs):
                return f(g(*args, **kwargs))

        else:
            raise TypeError(
                (
                    f"Could not compose step {f.__name__} with step {g.__name__} "
                    "due to return type signature mismatch: expected "
                    f"{f_in[0]}, got {g_out}."
                )
            )
    elif len(f_in) > 1:
        raise TypeError(
            (
                f"Could not compose step {f.__name__} with step {g.__name__} "
                f"{f.__name__} has more than 1 positional (required) argument, "
                "but only 1 would be given."
            )
        )
    else:
        raise TypeError(
            (
                f"Could not compose {f.__name__} with step {g.__name__}. "
                f"{f.__name__} takes 0 positional arguments, but would be called "
                f"on {g.__name__}'s return value."
            )
        )
    # give the returned function a new signature, corresponding
    # to g's input types and f's return type
    f_of_g.__signature__ = Signature(  # type: ignore
        parameters=list(g_sig.parameters.values()),
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


def read_conda_deps(conda_file: str) -> Tuple[Optional[str], List[str], List[str]]:
    """parse the dependencies contained in the given file and return as a (python_version, conda_dependencies, pip_dependencies) tuple"""

    try:
        with open(conda_file, "r") as f:
            contents = yaml.safe_load(f.read())
    except FileNotFoundError:
        logger.error("File not found:", conda_file)
        raise
    except PermissionError:
        logger.error("Insufficient permissions to read the file:", conda_file)
        raise
    except IOError as e:
        logger.error("An error occurred while reading the file:", e)
        raise
    except yaml.YAMLError as e:
        logger.error("An error occurred while parsing the YAML content:", e)
        raise

    if "dependencies" not in contents:
        logger.error(
            f"Parsed {conda_file} as {contents}, but did not find the expected"
            "'dependencies' field. This should not happen if using a"
            "conda-generated .yaml/.yml file."
        )
        raise KeyError

    python_spec_re = re.compile(r"python[=<>! ]*")
    pip_spec_re = re.compile(r"pip[=<>!]?")

    python_version = None
    conda_dependencies = []
    pip_dependencies = []

    for dependency in contents["dependencies"]:
        if isinstance(dependency, str):
            if python_spec_re.match(dependency):
                # keep the right hand side only
                rhs = re.sub(python_spec_re, "", dependency)
                python_version = rhs.strip() or python_version
            elif pip_spec_re.match(dependency):
                # we already know we need pip
                continue
            else:
                conda_dependencies += [dependency]
        elif isinstance(dependency, dict):
            # pip dependencies are already a flat list if they exist
            pip_dependencies += dependency.get("pip", [])

    python_version = python_version or ".".join(map(str, sys.version_info[:3]))
    pip_dependencies = validate_pip_lines(pip_dependencies)
    return python_version, conda_dependencies, pip_dependencies


def validate_pip_lines(lines: List[str]) -> List[str]:
    """given a list of pip requirements (e.g. the non-comment lines of a requirements.txt file),
    validate them according to the spec at https://peps.python.org/pep-0508/.
    """
    # NOTE: this is significantly stricter than pip's actual behavior.  pip
    # powerusers might be surprised by e.g. inline options such as --hash or
    # --index-url not being valid, so we might want to re-think this if we see
    # users consistently running into a wall here.
    #
    # real example: pip could install (from cli or as a line in requirements.txt)
    # `git+https://github.com/exalearn/ExaMol.git` by itself if asked to, but it
    # isn't PEP-508 without `examol @ ...` prefixed to it, so this would fail to
    # parse
    requirements = []
    for line in lines:
        try:
            r = Requirement(line)
            requirements += [r]
        except InvalidRequirement:
            logger.warning(f"Could not parse requirement line: {line}")
            raise

    return [str(r) for r in requirements]


def clean_identifier(name: str) -> str:
    """Clean the name provided for use as a pipeline's python identifier."""
    orig = name
    # Remove invalid characters, replacing with _
    name = re.sub("[^0-9a-zA-Z_]", "_", name)

    # Remove leading characters until we find a letter
    name = re.sub("^[^a-zA-Z]+", "", name)

    # Remove doubled/trailing underscores
    name = re.sub("__+", "_", name).strip("_")

    if not name:
        # name consisted only of invalid characters
        raise ValueError(
            "Invalid short_name. This argument should contain a valid python identifier "
            "(i.e. something usable as a variable name)."
        )

    # truncate to sane length, though not strictly necessary
    name = name[:50]

    if iskeyword(name):
        name += "_"

    if name != orig:
        logger.info(f'Generated valid short_name "{name}" from "{orig}".')

    return name.lower()
