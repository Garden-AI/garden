from pathlib import Path

from garden_ai import GardenClient, Pipeline, step

client = GardenClient()


@step
def run_inference(
    input_arg: object,
) -> object:
    import os

    result = dict(os.environ)

    return result


# the step functions will be composed in order by the pipeline:
ALL_STEPS = (run_inference,)

REQUIREMENTS_FILE = (Path(__file__).parent / "requirements.txt").resolve()

environment_variable_predictor: Pipeline = client.create_pipeline(
    title="environment variable predictor",
    steps=ALL_STEPS,
    requirements_file=str(REQUIREMENTS_FILE),
    authors=["Owen"],
    contributors=[],
    description="tests env vars in registered pipelines",
    version="0.0.1",
    year=2023,
    tags=[]
)
