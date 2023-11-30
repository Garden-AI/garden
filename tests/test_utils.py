import pytest

from garden_ai.utils.misc import extract_email_from_globus_jwt
from garden_ai.utils.notebooks import clear_cells, is_over_size_limit


def test_extract_email_from_globus_jwt_happy_path(identity_jwt):
    assert extract_email_from_globus_jwt(identity_jwt) == "willengler@uchicago.edu"


def test_extract_email_from_globus_jwt_malformed(identity_jwt):
    jwt_segments = identity_jwt.split(".")
    jwt_segments[1] += "asdfasd"
    malformed_jwt = ".".join(jwt_segments)
    with pytest.raises(Exception):
        extract_email_from_globus_jwt(malformed_jwt)


notebook_with_outputs = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [
                {
                    "data": {"text/plain": ["Hello"]},
                    "execution_count": 1,
                    "metadata": {},
                    "output_type": "execute_result",
                }
            ],
            "source": ["print('Hello')"],
        }
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 4,
}
cleared_notebook = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": ["print('Hello')"],
        }
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 4,
}


def test_clear_cells():
    assert clear_cells(notebook_with_outputs) == cleared_notebook


def test_is_over_size_limit():
    assert not is_over_size_limit(cleared_notebook)

    # 6MB worth of "a"s
    big_notebook = cleared_notebook.copy()
    big_notebook["cells"][0]["source"] = ["a" * 1024 * 1024 * 6]
    assert is_over_size_limit(big_notebook)
