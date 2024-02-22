import base64
import json
import logging
import re
import functools
from keyword import iskeyword

import requests
from typing_extensions import TypeAlias

JSON: TypeAlias = str

logger = logging.getLogger()


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


def clean_identifier(name: str) -> str:
    """Clean the name provided for use as an entrypoint's python identifier."""
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


def trackcalls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # note: attribute set only after func completes execution
        # so that func itself can determine if it's been called
        try:
            return func(*args, **kwargs)
        finally:
            wrapper.has_been_called = True

    wrapper.has_been_called = False
    return wrapper
