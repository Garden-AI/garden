import json
import os
from pathlib import Path

import ipywidgets as widgets  # type: ignore
import nbformat
import pytest
from garden_ai.constants import GardenConstants
from garden_ai.notebook_metadata import (
    NOTEBOOK_DISPLAY_METADATA_CELL,
    RequirementsData,
    _has_metadata_cell_tag,
    add_notebook_metadata,
    display_metadata_widget,
    get_notebook_metadata,
    read_requirements_data,
    save_requirements_data,
    set_notebook_metadata,
)


def test_add_metadata(
    mocker,
    notebook_empty,
    notebook_with_empty_metadata,
):
    ntbk = nbformat.from_dict(notebook_empty)

    mocker.patch("garden_ai.notebook_metadata._read_notebook", return_value=ntbk)

    nbformat_write_mock = mocker.patch("garden_ai.notebook_metadata.nbformat.write")

    add_notebook_metadata(None)

    write_arg = nbformat_write_mock.call_args.args[0]
    assert write_arg == notebook_with_empty_metadata


def test_get_metadata(
    mocker,
    notebook_metadata_pip,
    notebook_with_metadata,
):
    ntbk = nbformat.from_dict(notebook_with_metadata)

    mocker.patch("garden_ai.notebook_metadata._read_notebook", return_value=ntbk)

    notebook_metadata = get_notebook_metadata(None).model_dump()

    assert notebook_metadata == notebook_metadata_pip


def test_set_metadata(
    mocker,
    notebook_with_metadata,
    notebook_with_empty_metadata,
):
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


def test_read_requirements_data_pip(
    mocker,
    pip_requirements_raw,
    pip_requirements,
):
    # pip file
    mocker.patch("builtins.open", mocker.mock_open(read_data=pip_requirements_raw))
    requirements_data = read_requirements_data(Path("file.txt"))
    assert requirements_data.model_dump() == pip_requirements


def test_write_requirements(
    mocker,
    pip_requirements,
    pip_requirements_raw,
):
    mock_file = mocker.patch("builtins.open", mocker.mock_open())
    save_requirements_data(Path("file.txt"), RequirementsData(**pip_requirements))
    mock_file().write.assert_called_with(pip_requirements_raw)


def test_read_requirements_data_returns_none_if_invalid_format(
    tmp_path,
):
    # An invalid file format
    file = tmp_path / "some_file.json"
    requirements_data = read_requirements_data(file)
    assert requirements_data is None


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


def test_get_metadata_returns_empty_metadata_if_garden_metadata_not_found(
    tmp_notebook_empty,
):
    ntbk_meta = get_notebook_metadata(tmp_notebook_empty)
    assert ntbk_meta.global_notebook_doi is None
    assert ntbk_meta.notebook_image_name is None
    assert ntbk_meta.notebook_image_uri is None
    assert ntbk_meta.notebook_requirements == RequirementsData(
        file_format="pip", contents=[]
    )


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
    assert ntbk_meta.notebook_requirements == RequirementsData(
        file_format="pip", contents=[]
    )


def test_get_metadata_returns_metadata_if_already_present(
    notebook_metadata_pip,
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


def test_display_metadata_widget_displays(
    mocker,
    patch_notebook_env,
    patch_get_notebook_metadata,
):
    mock_display = mocker.patch(
        "garden_ai.notebook_metadata.display",
    )

    display_metadata_widget()

    mock_display.assert_called_once()


def test_change_to_doi_updates_notebook_metadata_json(
    mocker,
    patch_notebook_env,
    patch_get_notebook_metadata,
):
    # Create a stand-in for the doi_widget and patch it in
    doi_widget = widgets.Textarea("Some DOI")
    mocker.patch(
        "garden_ai.notebook_metadata._build_doi_widget",
        return_value=doi_widget,
    )

    # Display the metadata widget to kick everything off
    display_metadata_widget()

    # Update the doi widget simulating a user interaction
    # This should cause a new notebook_metadata.json to be written
    doi_widget.value = "New DOI"

    # Grab the json file
    ntbk_meta_path = (
        Path(os.environ.get("NOTEBOOK_PATH")).parent / "notebook_metadata.json"
    )
    assert ntbk_meta_path.exists()
    with open(ntbk_meta_path, "r") as f:
        metadata = json.load(f)

    # See if the updated doi is correct
    assert metadata["global_notebook_doi"] == doi_widget.value


def test_change_to_base_image_updates_notebook_metadata_json(
    mocker,
    patch_notebook_env,
    patch_get_notebook_metadata,
):
    # Create a stand-in for the base_image_widget and patch it in
    base_image_widget = widgets.Dropdown(
        options=GardenConstants.PREMADE_IMAGES.keys(), value="3.10-base", disabled=False
    )
    mocker.patch(
        "garden_ai.notebook_metadata._build_base_image_widget",
        return_value=base_image_widget,
    )

    # Display the metadata widget to kick everything off
    display_metadata_widget()

    # Update the base image widget simulating a user interaction
    # This should cause a new notebook_metadata.json to be written
    base_image_widget.value = "3.11-base"

    # Grab the json file
    ntbk_meta_path = (
        Path(os.environ.get("NOTEBOOK_PATH")).parent / "notebook_metadata.json"
    )
    assert ntbk_meta_path.exists()
    with open(ntbk_meta_path, "r") as f:
        metadata = json.load(f)

    # See if the updated base image is correct
    assert metadata["notebook_image_name"] == base_image_widget.value


def test_change_to_requirements_updates_notebook_metadata_json(
    mocker,
    patch_notebook_env,
    patch_get_notebook_metadata,
):
    # Create a stand-in for the reqs_widget and patch it in
    reqs_widget = widgets.Textarea(
        value="some-package",
        placeholder="Requirements",
        layout=widgets.Layout(width="100%", height="80px"),
        continuous_update=False,
        disabled=False,
    )
    mocker.patch(
        "garden_ai.notebook_metadata._build_reqs_widget",
        return_value=reqs_widget,
    )

    # Display the metadata widget to kick everything off
    display_metadata_widget()

    # Update the reqs widget simulating a user interaction
    # This should cause a new notebook_metadata.json to be written
    reqs_widget.value = "new-package"

    # Grab the json file
    ntbk_meta_path = (
        Path(os.environ.get("NOTEBOOK_PATH")).parent / "notebook_metadata.json"
    )
    assert ntbk_meta_path.exists()
    with open(ntbk_meta_path, "r") as f:
        metadata = json.load(f)

    # See if the updated reqs are correct
    assert reqs_widget.value in metadata["notebook_requirements"]["contents"]
