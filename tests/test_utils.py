import pytest

from garden_ai.utils import extract_email_from_globus_jwt


def test_extract_email_from_globus_jwt_happy_path(identity_jwt):
    assert extract_email_from_globus_jwt(identity_jwt) == 'willengler@uchicago.edu'


def test_extract_email_from_globus_jwt_malformed(identity_jwt):
    jwt_segments = identity_jwt.split('.')
    jwt_segments[1] += 'asdfasd'
    malformed_jwt = '.'.join(jwt_segments)
    with pytest.raises(Exception):
        extract_email_from_globus_jwt(malformed_jwt)
