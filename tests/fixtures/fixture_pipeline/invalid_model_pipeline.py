## THIS FILE WAS AUTOMATICALLY GENERATED ##
from garden_ai import GardenClient, Model, Pipeline, step
import typing

##################################### STEPS #####################################
"""
Brief notes on steps (see docs for more detail):

    - you may define your pipeline using as many or as few @steps as you like.

    - Any python function or callable can be made into a step by decorating it
        with `@step`, like below.

    - these functions will be composed in the pipeline (i.e. calling the pipeline
        is equivalent to calling each step in order).

    - the steps MUST have valid type-hints for all positional arguments and
        return types.
      - don't use `Any` or `None` in step annotations
      - these type-hints are used to verify that steps are compatible when
          composing (no checking at runtime)
"""


# example step using the decorator:
@step
def preprocessing_step(input_data: object) -> object:
    """ """
    # TODO
    pass


@step()
def another_step(data: object) -> object:
    # TODO
    pass


@step
def run_inference(
    input_arg: object,
    model=Model(
        "this is not a model name, no one will name a model this, it is for tests"
    ),
) -> object:
    pass


# the step functions will be composed in order by the pipeline:
ALL_STEPS = (
    preprocessing_step,
    another_step,
    run_inference,
)

REQUIREMENTS_FILE = None  # to specify additional dependencies, replace `None`
# with an "/absolute/path/to/requirements.txt"

################################### PIPELINE ####################################

fixture_pipeline = Pipeline(
    title="Invalid model fixture pipeline",
    steps=ALL_STEPS,
    requirements_file=REQUIREMENTS_FILE,
    authors=["Garden Team"],
    contributors=[],
    description="",
    version="0.0.1",
    year=2023,
    tags=[],
    uuid="b537520b-e86e-45bf-8566-4555a72b0b08",  # WARNING: DO NOT EDIT UUID
)
