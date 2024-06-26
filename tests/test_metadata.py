import nbformat
from pathlib import Path

from unittest.mock import mock_open, patch

from garden_ai.notebook_metadata import (
    add_notebook_metadata,
    set_notebook_metadata,
    get_notebook_metadata,
    read_requirements_data,
    save_requirements_data,
    RequirementsData,
    NOTEBOOK_DISPLAY_METADATA_CELL,
)


pip_requirements_raw = "scikit-learn==1.2.2\npandas\n"
pip_requirements = {"file_format": "pip", "contents": ["scikit-learn==1.2.2", "pandas"]}

conda_requirements_raw = "name: garden-test\ndependencies:\n- python=3.9\n- pip\n- pip:\n  - scikit-learn==1.2.2\n  - pandas\n"
conda_requirements = {
    "file_format": "conda",
    "contents": {
        "name": "garden-test",
        "dependencies": [
            "python=3.9",
            "pip",
            {"pip": ["scikit-learn==1.2.2", "pandas"]},
        ],
    },
}

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
notebook_metadata_conda = {
    "global_notebook_doi": "10.23677/testdoi",
    "notebook_image_name": "3.9-base",
    "notebook_requirements": conda_requirements,
    "notebook_image_uri": "A_BASE_IMAGE_URI",
}


def test_add_metadata(mocker):
    ntbk = nbformat.from_dict(notebook_empty)

    mocker.patch("garden_ai.notebook_metadata._read_notebook", return_value=ntbk)

    nbformat_write_mock = mocker.patch("garden_ai.notebook_metadata.nbformat.write")

    add_notebook_metadata(None)

    write_arg = nbformat_write_mock.call_args.args[0]

    assert write_arg == notebook_with_empty_metadata


def test_get_metadata(mocker):
    ntbk = nbformat.from_dict(notebook_with_metadata)

    mocker.patch("garden_ai.notebook_metadata._read_notebook", return_value=ntbk)

    notebook_metadata = get_notebook_metadata(None).model_dump()

    assert notebook_metadata == notebook_metadata_pip


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


def test_read_requirements(mocker):
    # pip file
    with patch("builtins.open", mock_open(read_data=pip_requirements_raw)):
        requirements_data = read_requirements_data(Path("file.txt"))
        assert requirements_data.model_dump() == pip_requirements

    # conda file
    with patch("builtins.open", mock_open(read_data=conda_requirements_raw)):
        requirements_data = read_requirements_data(Path("file.yaml"))
        assert requirements_data.model_dump() == conda_requirements


def test_write_requirements(mocker):
    with patch("builtins.open", mock_open()) as mock_file:
        save_requirements_data(Path("file.txt"), RequirementsData(**pip_requirements))
        mock_file().write.assert_called_with(pip_requirements_raw)

    with patch("builtins.open", mock_open()) as mock_file:
        yaml_dump_mock = mocker.patch(
            "garden_ai.notebook_metadata.yaml.dump", return_value=True
        )
        save_requirements_data(
            Path("file.yaml"), RequirementsData(**conda_requirements)
        )
        write_arg = yaml_dump_mock.call_args.args[0]

        assert conda_requirements["contents"] == write_arg
