from copy import deepcopy
import pytest  # noqa: F401

from garden_ai.utils.notebooks import (
    clear_cells,
    is_over_size_limit,
    generate_botanical_filename,
)

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
    big_notebook = deepcopy(cleared_notebook)
    big_notebook["cells"][0]["source"] = ["a" * 1024 * 1024 * 6]
    assert is_over_size_limit(big_notebook)


def test_generate_botanical_name_returns_name_with_ipynb_extension():
    name = generate_botanical_filename()
    assert name.endswith(".ipynb")
