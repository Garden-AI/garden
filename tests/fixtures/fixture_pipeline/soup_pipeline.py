from pathlib import Path
from typing import List

from garden_ai import GardenClient, Model, Pipeline, step

client = GardenClient()


class Soup:
    def __init__(self, ingredients):
        self.contents = [float(ingredient) for ingredient in ingredients]


@step
def split_peas(ps: List) -> List[tuple]:
    return [(p / 2, p / 2) for p in ps]


@step
def make_soup(splits: List[tuple]) -> Soup:
    return Soup(x for x, _ in splits)


@step
def rate_soup(
    soup_sample: Soup,
    model=Model("owenpriceskelly@uchicago.edu-toy-model/1"),
) -> float:
    """huh, wonder what this Model is for?"""

    assert model.model is None  # lazy - not loaded yet
    try:
        model.predict(soup_sample)
    except Exception:
        assert model.model is not None  # loaded, whatever it was
    return 10 / 10


# the step functions will be composed in order by the pipeline:
ALL_STEPS = (
    split_peas,
    make_soup,
    rate_soup,
)

REQUIREMENTS_FILE = str((Path(__file__).parent / "requirements.txt").resolve())


soup_edibility_predictor: Pipeline = client.create_pipeline(
    title="soup edibility predictor",
    doi="10.26311/fake-doi",
    steps=ALL_STEPS,
    requirements_file=REQUIREMENTS_FILE,
    authors=["Me"],
    contributors=["You"],
    description="Big Pea Data as in big peas, not like big data",
    version="0.0.1",
    year=2023,
    tags=["you're it"]
)
