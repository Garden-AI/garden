from pathlib import Path

import nbformat
import pytest

from garden_ai.notebook_metadata import (
    add_notebook_metadata,
    set_notebook_metadata,
    get_notebook_metadata,
    read_requirements_data,
    save_requirements_data,
    RequirementsData,
    NOTEBOOK_DISPLAY_METADATA_CELL,
    _has_metadata_cell_tag,
)


pip_requirements_raw = "scikit-learn==1.2.2\npandas\n"
pip_requirements = {"file_format": "pip", "contents": ["scikit-learn==1.2.2", "pandas"]}

notebook_empty = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 4}
notebook_with_empty_metadata = {
    "cells": [
        {
            "cell_type": "code",
            "metadata": {
                "tags": ["garden_display_metadata_cell"],
            },
            "execution_count": None,
            "source": NOTEBOOK_DISPLAY_METADATA_CELL,
            "outputs": [],
        }
    ],
    "metadata": {
        "garden_metadata": {
            "notebook_image_uri": None,
            "global_notebook_doi": None,
            "notebook_image_name": None,
            "notebook_requirements": None,
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
notebook_with_metadata = {
    "cells": [
        {
            "cell_type": "code",
            "metadata": {
                "tags": ["garden_display_metadata_cell"],
            },
            "execution_count": None,
            "source": NOTEBOOK_DISPLAY_METADATA_CELL,
            "outputs": [],
        }
    ],
    "metadata": {
        "garden_metadata": {
            "notebook_image_uri": "A_BASE_IMAGE_URI",
            "global_notebook_doi": "10.23677/testdoi",
            "notebook_image_name": "3.9-base",
            "notebook_requirements": pip_requirements,
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

notebook_metadata_pip = {
    "global_notebook_doi": "10.23677/testdoi",
    "notebook_image_name": "3.9-base",
    "notebook_requirements": pip_requirements,
    "notebook_image_uri": "A_BASE_IMAGE_URI",
}


@pytest.mark.integration
def test_add_notebook_metadata_writes_metadata_cell_to_notebook(
    tmp_notebook_empty,
):
    # A new notebook should not have the metadata cell
    ntbk = nbformat.read(tmp_notebook_empty, nbformat.NO_CONVERT)
    assert not _has_metadata_cell_tag(ntbk)

    # Add the metadata cell
    add_notebook_metadata(tmp_notebook_empty)

    # The notebook should now have the metadata cell as its first cell
    ntbk = nbformat.read(tmp_notebook_empty, nbformat.NO_CONVERT)
    assert _has_metadata_cell_tag(ntbk)
    assert ntbk.cells[0].source == NOTEBOOK_DISPLAY_METADATA_CELL


@pytest.mark.integration
def test_get_metadata_returns_empty_metadata_if_garden_metadata_not_foune(
    tmp_notebook_empty,
):
    ntbk_meta = get_notebook_metadata(tmp_notebook_empty)
    assert ntbk_meta.global_notebook_doi is None
    assert ntbk_meta.notebook_image_name is None
    assert ntbk_meta.notebook_image_uri is None
    assert ntbk_meta.notebook_requirements is None


@pytest.mark.integration
def test_get_metadata_returns_empty_metadata_if_garden_metadata_is_bad(
    tmp_notebook_empty,
):
    # Set up a bad garden_metadata cell
    ntbk = nbformat.read(tmp_notebook_empty, nbformat.NO_CONVERT)
    ntbk["metadata"]["garden_metadata"] = {"some": "bad_data"}
    nbformat.write(ntbk, tmp_notebook_empty, version=nbformat.NO_CONVERT)

    # Get the metadata
    ntbk_meta = get_notebook_metadata(tmp_notebook_empty)
    assert ntbk_meta.global_notebook_doi is None
    assert ntbk_meta.notebook_image_name is None
    assert ntbk_meta.notebook_image_uri is None
    assert ntbk_meta.notebook_requirements is None


@pytest.mark.integration
def test_get_metadata_returns_metadata_if_already_present(
    tmp_notebook_empty,
):
    # Set up a garden_metadata cell
    ntbk = nbformat.read(tmp_notebook_empty, nbformat.NO_CONVERT)
    ntbk["metadata"]["garden_metadata"] = notebook_metadata_pip
    nbformat.write(ntbk, tmp_notebook_empty, version=nbformat.NO_CONVERT)

    # Get the metadata
    ntbk_meta = get_notebook_metadata(tmp_notebook_empty)
    assert ntbk_meta.global_notebook_doi == notebook_metadata_pip["global_notebook_doi"]
    assert ntbk_meta.notebook_image_name == notebook_metadata_pip["notebook_image_name"]
    assert ntbk_meta.notebook_image_uri == notebook_metadata_pip["notebook_image_uri"]
    assert ntbk_meta.notebook_requirements == RequirementsData(
        **notebook_metadata_pip["notebook_requirements"]
    )


def test_set_metadata(mocker):
    ntbk = nbformat.from_dict(notebook_with_empty_metadata)

    mocker.patch("garden_ai.notebook_metadata._read_notebook", return_value=ntbk)
    nbformat_write_mock = mocker.patch("garden_ai.notebook_metadata.nbformat.write")

    set_notebook_metadata(
        None,
        "10.23677/testdoi",
        "3.9-base",
        "A_BASE_IMAGE_URI",
        RequirementsData(file_format="pip", contents=["scikit-learn==1.2.2", "pandas"]),
    )

    write_arg = nbformat_write_mock.call_args.args[0]
    assert write_arg == notebook_with_metadata


@pytest.mark.integration
def test_set_metadata_raises_if_unable_to_parse_notebook(
    tmp_path,
):
    bad_notebook = tmp_path / "bad_notebook.ipynb"
    with pytest.raises(Exception):
        set_notebook_metadata(
            bad_notebook,
            "10.23677/testdoi",
            "3.9-base",
            "A_BASE_IMAGE_URI",
            RequirementsData(
                file_format="pip", contents=["scikit-learn==1.2.2", "pandas"]
            ),
        )


def test_read_requirements_data_pip(mocker):
    # pip file
    mocker.patch("builtins.open", mocker.mock_open(read_data=pip_requirements_raw))
    requirements_data = read_requirements_data(Path("file.txt"))
    assert requirements_data.model_dump() == pip_requirements


def test_read_requirements_data_returns_none_if_invalid_format(
    tmp_path,
):
    # An invalid file format
    file = tmp_path / "some_file.json"
    requirements_data = read_requirements_data(file)
    assert requirements_data is None


def test_write_requirements(mocker):
    mock_file = mocker.patch("builtins.open", mocker.mock_open())
    save_requirements_data(Path("file.txt"), RequirementsData(**pip_requirements))
    mock_file().write.assert_called_with(pip_requirements_raw)
