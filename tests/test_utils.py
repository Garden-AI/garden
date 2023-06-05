import pytest

from garden_ai.utils.misc import (
    InvalidRequirement,
    extract_email_from_globus_jwt,
    validate_pip_lines,
)


def test_extract_email_from_globus_jwt_happy_path(identity_jwt):
    assert extract_email_from_globus_jwt(identity_jwt) == "willengler@uchicago.edu"


def test_extract_email_from_globus_jwt_malformed(identity_jwt):
    jwt_segments = identity_jwt.split(".")
    jwt_segments[1] += "asdfasd"
    malformed_jwt = ".".join(jwt_segments)
    with pytest.raises(Exception):
        extract_email_from_globus_jwt(malformed_jwt)


def test_validate_pip_lines():
    # Valid requirements
    valid_lines = [
        "numpy==1.21.2",
        "pandas>=1.3.3",
        "scipy!=1.7.2",
        "matplotlib",
        "package@ https://github.com/user/package/archive/v1.0.0.tar.gz",
    ]

    result = validate_pip_lines(valid_lines)
    assert set(result) == set(valid_lines)

    with pytest.raises(InvalidRequirement):
        invalid_lines = [
            "numpy --hash=sha256:abc",
            "https://github.com/user/package/archive/v1.0.0.tar.gz",
        ]
        result = validate_pip_lines(valid_lines + invalid_lines)
