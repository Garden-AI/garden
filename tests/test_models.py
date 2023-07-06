import sys
import json
from typing import Any, Iterable, List, Tuple, Union

import pytest
from pydantic import ValidationError

from garden_ai import Garden, Pipeline, RegisteredPipeline, Step, local_data, step


def test_create_empty_garden(garden_client):
    # feels silly, but we do want users to be able to initialize an empty garden
    # & fill in required stuff later

    # object should exist with default-illegal fields
    garden = garden_client.create_garden()

    assert not garden.authors
    assert not garden.title


def test_validate_all_fields(garden_all_fields):
    garden_all_fields.validate()


def test_garden_datacite(garden_title_authors_doi_only):
    data = json.loads(garden_title_authors_doi_only.datacite_json())

    assert isinstance(data["creators"], list)
    assert isinstance(data["titles"], list)
    assert data["publisher"] == "thegardens.ai"


def test_pipeline_datacite(garden_client):
    @step
    def four() -> int:
        return 4

    pipeline = garden_client.create_pipeline(
        authors=["Team, Garden"], title="Lorem Ipsum", steps=(four,)
    )
    data = json.loads(pipeline.datacite_json())

    assert isinstance(data["creators"], list)
    assert isinstance(data["titles"], list)
    assert data["publisher"] == "thegardens.ai"


def test_validate_no_fields(garden_no_fields):
    with pytest.raises(ValidationError):
        garden_no_fields.validate()


def test_validate_required_only(garden_no_fields):
    garden = garden_no_fields
    garden.authors = ["Mendel, Gregor"]
    with pytest.raises(ValidationError):
        garden.validate()
    garden.title = "Experiments on Plant Hybridization"
    garden.validate()


def test_garden_can_access_pipeline_as_attribute(
    mocker, database_with_connected_pipeline
):
    mocker.patch(
        "garden_ai.local_data.LOCAL_STORAGE", new=database_with_connected_pipeline
    )
    garden = local_data.get_local_garden_by_doi("10.23677/fake-doi")
    assert isinstance(garden.fixture_pipeline, RegisteredPipeline)


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


def test_steps_collect_source():
    with pytest.raises(ValidationError):
        # won't be able to find source (or annotations) for builtins
        builtin_sum_step = Step(func=sum)

    # ... but this is fine
    @step
    def builtin_sum_step(xs: Iterable) -> Union[int, float]:
        return sum(xs)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_pipeline_compose_union(tmp_requirements_txt):
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
        requirements_file=str(tmp_requirements_txt),
        title="good pipeline",
        steps=[str_only, wants_int_or_str],
        doi="10.26311/fake-doi",
    )

    also_good = Pipeline(  # noqa: F841
        authors=["mendel"],
        requirements_file=str(tmp_requirements_txt),
        title="good pipeline",
        steps=[str_only, wants_int_or_str_old_syntax],
        doi="10.26311/fake-doi",
    )

    union_order_doesnt_matter = Pipeline(  # noqa: F841
        authors=["mendel"],
        requirements_file=str(tmp_requirements_txt),
        title="good pipeline",
        steps=[wants_int_or_str, wants_int_or_str],
        doi="10.26311/fake-doi",
    )

    union_syntax_doesnt_matter = Pipeline(  # noqa: F841
        authors=["mendel"],
        requirements_file=str(tmp_requirements_txt),
        title="good pipeline",
        steps=[wants_int_or_str, wants_int_or_str_old_syntax],
        doi="10.26311/fake-doi",
    )

    with pytest.raises(ValidationError):
        union_str_does_not_subtype_str = Pipeline(  # noqa: F841
            authors=["mendel"],
            requirements_file=str(tmp_requirements_txt),
            title="bad pipeline",
            steps=[wants_int_or_str, str_only],
            doi="10.26311/fake-doi",
        )

    with pytest.raises(ValidationError):
        old_union_str_does_not_subtype_str = Pipeline(  # noqa: F841
            authors=["mendel"],
            requirements_file=str(tmp_requirements_txt),
            title="bad pipeline",
            steps=[wants_int_or_str_old_syntax, str_only],
            doi="10.26311/fake-doi",
        )
    return


def test_step_authors_are_pipeline_contributors(pipeline_toy_example):
    pipe = pipeline_toy_example
    for s in pipe.steps:
        for author in s.authors:
            assert author in pipe.contributors
        for contributor in s.contributors:
            assert contributor in pipe.contributors


def test_step_collect_model(step_with_model):
    assert step_with_model.model_full_names == ["email@addr.ess/fake-model"]


def test_pipeline_collects_own_requirements(
    pipeline_using_step_with_model, tmp_requirements_txt
):
    with open(tmp_requirements_txt, "r") as f:
        contents = f.read()
        for dependency in pipeline_using_step_with_model.pip_dependencies:
            assert dependency in contents or dependency == "garden-ai==0.0.0"

    assert "python=" not in "".join(pipeline_using_step_with_model.conda_dependencies)


def test_pipeline_collects_step_models(pipeline_using_step_with_model):
    assert pipeline_using_step_with_model.model_full_names == [
        "email@addr.ess/fake-model"
    ]


def test_step_compose_ignores_defaults(tmp_requirements_txt):
    @step
    def returns_tuple(a: int, b: str) -> Tuple[int, str]:
        pass

    @step
    def wants_tuple_ignoring_default(arg1: Tuple[int, str], x: List = []) -> float:
        pass

    @step
    def wants_tuple_as_args_ignoring_default(
        arg1: int, arg2: str, x: List = []
    ) -> float:
        pass

    good = Pipeline(  # noqa: F841
        authors=["mendel"],
        requirements_file=str(tmp_requirements_txt),
        title="composes tuple-as-tuple w/ default",
        steps=[returns_tuple, wants_tuple_ignoring_default],
        doi="10.26311/fake-doi",
    )
