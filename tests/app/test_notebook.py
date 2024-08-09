import pytest

from garden_ai import GardenConstants


@pytest.mark.cli
def test_no_args_prints_usage(
    cli_runner,
    app,
):
    cli_args = [
        "notebook",
    ]

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    assert "Usage:" in result.output


@pytest.mark.cli
def test_list_premade_images(
    cli_runner,
    app,
):
    cli_args = [
        "notebook",
        "list-premade-images",
    ]

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0

    for image in GardenConstants.PREMADE_IMAGES.keys():
        assert image in result.output


@pytest.mark.cli
def test_start_no_args_prints_usage(
    cli_runner,
    app,
):
    cli_args = [
        "notebook",
        "start",
    ]

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    assert "Usage:" in result.output


@pytest.mark.cli
def test_start_confirms_notebook_path(
    cli_runner,
    app,
):
    notebook_path = "test_notebook.ipynb"
    image = list(GardenConstants.PREMADE_IMAGES.keys())[0]
    cli_args = ["notebook", "start", notebook_path, f"--base-image={image}"]

    result = cli_runner.invoke(
        app,
        cli_args,
        input="n\n",  # Say 'n' so we don't actually create the notebook
    )

    assert result.exit_code == 1
    assert notebook_path in result.output
    assert "Do you want to proceed?" in result.output


@pytest.mark.cli
def test_start_requires_base_image(
    cli_runner,
    app,
):
    notebook_path = "test_notebook.ipynb"
    cli_args = [
        "notebook",
        "start",
        notebook_path,
        # no base-image or custom-image
    ]

    result = cli_runner.invoke(
        app,
        cli_args,
    )
    assert "Please specify a base image" in result.output


@pytest.mark.cli
def test_start_rejects_base_AND_custom_image(
    cli_runner,
    app,
):
    notebook_path = "test_notebook.ipynb"
    base_image = list(GardenConstants.PREMADE_IMAGES.keys())[0]
    custom_image = "localhost:my/image"
    cli_args = [
        "notebook",
        "start",
        notebook_path,
        # Both base-image and custom-image
        f"--base-image={base_image}",
        f"--custom-image={custom_image}",
    ]

    result = cli_runner.invoke(
        app,
        cli_args,
    )
    assert result.exit_code == 1
    assert "Please specify only one" in result.output


@pytest.mark.cli
def test_start_rejects_invalid_base_image(
    cli_runner,
    app,
):
    notebook_path = "test_notebook.ipynb"
    unknown_base_image = "some-unknown/image"
    cli_args = [
        "notebook",
        "start",
        notebook_path,
        f"--base-image={unknown_base_image}",
    ]

    result = cli_runner.invoke(
        app,
        cli_args,
    )
    assert result.exit_code == 1
    assert (
        f"({unknown_base_image}) is not one of the Garden base images" in result.output
    )


@pytest.mark.cli
def test_publish_prints_usage_when_no_args(
    cli_runner,
    app,
):
    cli_args = [
        "notebook",
        "publish",
    ]

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 0
    assert "Usage:" in result.output


@pytest.mark.cli
def test_publish_rejects_non_ipynb_files(
    cli_runner,
    app,
):
    bad_path = "some_file.py"

    cli_args = [
        "notebook",
        "publish",
        f"{bad_path}",
    ]

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 1
    assert result.exception is not None


@pytest.mark.cli
def test_publish_rejects_non_existent_file(
    cli_runner,
    app,
):
    cli_args = [
        "notebook",
        "publish",
        "some_notebook_that_doesnt_exist.ipynb",
    ]

    result = cli_runner.invoke(app, cli_args)
    assert result.exit_code == 1
    assert result.exception is not None
