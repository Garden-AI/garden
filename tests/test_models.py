import os
import sys
from collections import namedtuple
from typing import Any, List, Tuple, Union

import pytest
from pydantic import ValidationError

from garden_ai import Garden, Pipeline, Step, step, RegisteredPipeline
from garden_ai import local_data
from garden_ai.mlmodel import upload_to_model_registry, LocalModel


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
    assert not garden.doi
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
    garden = local_data.get_local_garden_by_uuid("e1a3b50b-4efc-42c8-8422-644f4f858b87")
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
    )

    also_good = Pipeline(  # noqa: F841
        authors=["mendel"],
        requirements_file=str(tmp_requirements_txt),
        title="good pipeline",
        steps=[str_only, wants_int_or_str_old_syntax],
    )

    union_order_doesnt_matter = Pipeline(  # noqa: F841
        authors=["mendel"],
        requirements_file=str(tmp_requirements_txt),
        title="good pipeline",
        steps=[wants_int_or_str, wants_int_or_str],
    )

    union_syntax_doesnt_matter = Pipeline(  # noqa: F841
        authors=["mendel"],
        requirements_file=str(tmp_requirements_txt),
        title="good pipeline",
        steps=[wants_int_or_str, wants_int_or_str_old_syntax],
    )

    with pytest.raises(ValidationError):
        union_str_does_not_subtype_str = Pipeline(  # noqa: F841
            authors=["mendel"],
            requirements_file=str(tmp_requirements_txt),
            title="bad pipeline",
            steps=[wants_int_or_str, str_only],
        )

    with pytest.raises(ValidationError):
        old_union_str_does_not_subtype_str = Pipeline(  # noqa: F841
            authors=["mendel"],
            requirements_file=str(tmp_requirements_txt),
            title="bad pipeline",
            steps=[wants_int_or_str_old_syntax, str_only],
        )
    return


def test_pipeline_compose_tuple(tmp_requirements_txt):
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
        requirements_file=str(tmp_requirements_txt),
        title="good pipeline",
        steps=[returns_tuple, wants_tuple_as_tuple],
    )

    with pytest.raises(ValidationError):
        bad = Pipeline(  # noqa: F841
            authors=["mendel"],
            requirements_file=str(tmp_requirements_txt),
            title="backwards pipeline",
            steps=[wants_tuple_as_tuple, returns_tuple],
        )

    ugly = Pipeline(  # noqa: F841
        authors=["mendel"],
        requirements_file=str(tmp_requirements_txt),
        title="ugly (using *args) but allowed pipeline",
        steps=[returns_tuple, wants_tuple_as_args],
    )
    with pytest.raises(ValidationError):
        ugly_and_bad = Pipeline(  # noqa: F841
            authors=["mendel"],
            requirements_file=str(tmp_requirements_txt),
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


def test_upload_model(mocker, tmp_path):
    model_dir_path = tmp_path / "models"

    model_dir_path.mkdir(parents=True, exist_ok=True)
    model_path = model_dir_path / "model.pkl"
    model_path.write_text("abcd")

    # Prevents ML Flow from creating directory in /tests
    os.environ["MLFLOW_TRACKING_URI"] = str(tmp_path)

    MLFlowVersionResponse = namedtuple("MLFlowVersionResponse", "version")
    versions_response = [MLFlowVersionResponse("1"), MLFlowVersionResponse("0")]

    mocker.patch("pickle.load").return_value = "a deserialized model"
    mocker.patch("mlflow.sklearn.log_model")
    mocker.patch(
        "mlflow.tracking.MlflowClient.get_latest_versions"
    ).return_value = versions_response

    model_uri = "will@test.com-test_model/1"
    local_model = LocalModel(
        local_path=str(model_path),
        model_name="test_model",
        user_email="will@test.com",
        flavor="sklearn",
    )

    registered_model = upload_to_model_registry(local_model)
    assert registered_model.model_uri == model_uri
    assert registered_model.connections == []
    assert registered_model.version == "1"


def test_step_collect_model_requirements(step_with_model, tmp_conda_yml):
    # step should have collected these when it was initialized
    assert len(step_with_model.conda_dependencies) or len(
        step_with_model.pip_dependencies
    )

    with open(tmp_conda_yml, "r") as f:
        contents = f.read()
        for dep in step_with_model.pip_dependencies:
            assert dep in contents
    return


def test_step_collect_model(step_with_model):
    assert step_with_model.model_uris == ["email@addr.ess-fake-model/fake-version"]


def test_pipeline_collects_own_requirements(
    pipeline_using_step_with_model, tmp_requirements_txt
):
    with open(tmp_requirements_txt, "r") as f:
        contents = f.read()
        for dependency in pipeline_using_step_with_model.pip_dependencies:
            assert dependency in contents

    assert "python=" not in "".join(pipeline_using_step_with_model.conda_dependencies)


def test_pipeline_collects_step_requirements(
    pipeline_using_step_with_model, step_with_model
):
    for step_dependency in step_with_model.conda_dependencies:
        assert step_dependency in pipeline_using_step_with_model.conda_dependencies

    for step_dependency in step_with_model.pip_dependencies:
        assert step_dependency in pipeline_using_step_with_model.pip_dependencies


def test_pipeline_collects_step_models(pipeline_using_step_with_model):
    assert pipeline_using_step_with_model.model_uris == [
        "email@addr.ess-fake-model/fake-version"
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
    )

    ugly = Pipeline(  # noqa: F841
        authors=["mendel"],
        requirements_file=str(tmp_requirements_txt),
        title="composes tuple-as-*args w/ default",
        steps=[returns_tuple, wants_tuple_as_args_ignoring_default],
    )
