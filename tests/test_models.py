import sys
from typing import Tuple, Union, Any

import pytest
from garden_ai import Garden, Pipeline, Step, step
from pydantic import ValidationError


def test_create_empty_garden(garden_client):
    # feels silly, but we do want users to be able to initialize an empty garden
    # & fill in required stuff later

    # object should exist with default-illegal fields
    garden = garden_client.create_garden()

    assert not garden.authors
    assert not garden.title
    assert not garden.doi


def test_validate_all_fields(garden_all_fields):
    garden_all_fields.validate()


def test_validate_no_fields(garden_no_fields):
    with pytest.raises(ValidationError):
        garden_no_fields.validate()


def test_validate_required_only(garden_no_fields):
    garden = garden_no_fields
    garden.authors = ["Mendel, Gregor"]
    garden.title = "Experiments on Plant Hybridization"
    assert not garden.doi
    with pytest.raises(ValidationError):
        garden.validate()
    garden.doi = "10.26311/fake-doi"
    garden.validate()


def test_register_metadata(garden_client, garden_title_authors_doi_only, tmp_path):
    # uses pytest builtin fixture to write to tmp_path
    gc = garden_client
    garden = garden_title_authors_doi_only
    gc.register_metadata(garden, tmp_path)
    assert (tmp_path / f"{garden.uuid}.json").exists()
    with open(tmp_path / f"{garden.uuid}.json", "r") as f:
        json_contents = f.read()
        assert json_contents == garden.json()


def test_step_wrapper():
    # well-annotated callables are accepted; poorly annotated callables are not
    @step
    def well_annotated(a: int, b: str, g: Garden) -> Tuple[int, str, Garden]:
        pass

    assert isinstance(well_annotated, Step)

    with pytest.raises(ValidationError):

        @step
        def incomplete_annotated(a: int, b: str, g) -> int:
            pass


def test_step_disallow_anys():
    with pytest.raises(ValidationError):

        @step
        def any_arg_annotated(a: Any) -> object:
            pass

    with pytest.raises(ValidationError):

        @step
        def any_return_annotated(a: object) -> Any:
            pass


def test_auto_input_output_metadata():
    @step
    def well_annotated(a: int, b: str, g: Garden) -> Tuple[int, str, Garden]:
        pass

    assert (
        well_annotated.input_info
        == "{'a': <class 'int'>, 'b': <class 'str'>, 'g': <class 'garden_ai.gardens.Garden'>}"
    )
    assert (
        well_annotated.output_info
        == "return: typing.Tuple[int, str, garden_ai.gardens.Garden]"
    )

    @step(
        input_info="This step LOVES accepting arguments",
        output_info="it also returns important results",
    )
    def lovingly_annotated(a: int, b: str, g: Garden) -> Tuple[int, str, Garden]:
        pass

    assert lovingly_annotated.input_info == "This step LOVES accepting arguments"
    assert lovingly_annotated.output_info == "it also returns important results"


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_pipeline_compose_union():
    # check that pipelines can correctly compose tricky functions, which may be
    # annotated like typing.Union *or* like (A | B)
    @step
    def wants_int_or_str(arg: int | str) -> str | int:
        pass

    @step
    def wants_int_or_str_old_syntax(arg: Union[int, str]) -> Union[str, int]:
        pass

    @step
    def str_only(arg: str) -> str:
        pass

    good = Pipeline(  # noqa: F841
        authors=["mendel"],
        title="good pipeline",
        steps=[str_only, wants_int_or_str],
    )

    also_good = Pipeline(  # noqa: F841
        authors=["mendel"],
        title="good pipeline",
        steps=[str_only, wants_int_or_str_old_syntax],
    )

    union_order_doesnt_matter = Pipeline(  # noqa: F841
        authors=["mendel"],
        title="good pipeline",
        steps=[wants_int_or_str, wants_int_or_str],
    )

    union_syntax_doesnt_matter = Pipeline(  # noqa: F841
        authors=["mendel"],
        title="good pipeline",
        steps=[wants_int_or_str, wants_int_or_str_old_syntax],
    )

    with pytest.raises(ValidationError):
        union_str_does_not_subtype_str = Pipeline(  # noqa: F841
            authors=["mendel"],
            title="bad pipeline",
            steps=[wants_int_or_str, str_only],
        )

    with pytest.raises(ValidationError):
        old_union_str_does_not_subtype_str = Pipeline(  # noqa: F841
            authors=["mendel"],
            title="bad pipeline",
            steps=[wants_int_or_str_old_syntax, str_only],
        )
    return


def test_pipeline_compose_tuple():
    # check that pipelines can correctly compose tricky
    # functions which may or may not expect a plain tuple
    # to be treated as *args
    @step
    def returns_tuple(a: int, b: str) -> Tuple[int, str]:
        pass

    @step
    def wants_tuple_as_tuple(t: Tuple[int, str]) -> int:
        pass

    @step
    def wants_tuple_as_args(x: int, y: str) -> str:
        pass

    @step
    def wants_flipped_tuple_as_args(arg1: str, arg2: int) -> float:
        pass

    good = Pipeline(  # noqa: F841
        authors=["mendel"],
        title="good pipeline",
        steps=[returns_tuple, wants_tuple_as_tuple],
    )

    with pytest.raises(ValidationError):
        bad = Pipeline(  # noqa: F841
            authors=["mendel"],
            title="backwards pipeline",
            steps=[wants_tuple_as_tuple, returns_tuple],
        )

    ugly = Pipeline(  # noqa: F841
        authors=["mendel"],
        title="ugly (using *args) but allowed pipeline",
        steps=[returns_tuple, wants_tuple_as_args],
    )
    with pytest.raises(ValidationError):
        ugly_and_bad = Pipeline(  # noqa: F841
            authors=["mendel"],
            title="ugly (using *args) and disallowed pipeline",
            steps=[returns_tuple, wants_flipped_tuple_as_args],
        )


def test_step_authors_are_pipeline_contributors(pipeline_toy_example):
    pipe = pipeline_toy_example
    for s in pipe.steps:
        for author in s.authors:
            assert author in pipe.contributors
        for contributor in s.contributors:
            assert contributor in pipe.contributors


def test_pipeline_authors_are_garden_contributors(
    garden_all_fields, pipeline_toy_example
):
    # verify that adding a pipeline with new authors
    # updates the garden's 'contributors' field
    garden, pipe = garden_all_fields, pipeline_toy_example
    garden.add_new_pipeline(pipe.title, pipe.steps, authors=pipe.authors)

    known_authors = [a for a in garden.authors]
    known_contributors = [c for c in garden.contributors]
    garden._sync_author_metadata()
    # should never modify garden.authors implicitly
    assert known_authors == garden.authors
    # should not lose contributors by adding pipelines
    assert set(known_contributors) <= set(garden.contributors)
    # 'contributors' at any pipeline level should propagate to garden
    assert set(pipe.contributors) <= set(garden.contributors)
