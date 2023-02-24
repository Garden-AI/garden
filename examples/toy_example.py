from garden_ai import GardenClient, step, Pipeline, Garden
from typing import List


client = GardenClient()

# Create a new empty Garden with desired metadata
pea_garden: Garden = client.create_garden(
    authors=["Mendel, Gregor"],
    title="Experiments on Plant Hybridization",
    contributors=["St. Thomas Abbey"],
)
# metadata fields can also be set after object creation
pea_garden.year = "1863"
pea_garden.language = "en"
pea_garden.version = "0.0.1"
pea_garden.description = """
This Garden houses sophisticated ML pipelines for Big Pea Data extraction
and classification. It consists of a 2-hectare plot behind the monastery,
and a 30,000-plant dataset.
"""


# define a step using the decorator
@step(authors=["Sister Constance"])
def split_peas(ps: List) -> List[tuple]:
    return [(p / 2, p / 2) for p in ps]


# Fails due to incomplete annotations:
"""
@step(authors=["Friar Hugo"])
def make_soup(splits: List[tuple]):
    pass
  ...
ValidationError: 1 validation error for Step
func
    make_soup's definition is missing a return annotation, or returns None.
See also: https://peps.python.org/pep-0484/#type-definition-syntax (type=type_error)
"""


# Solution: non-builtin classes/ type annotations are supported
class Soup:  # noqa
    def __init__(self, ingredients):
        self.contents = [float(ingredient) for ingredient in ingredients]
        # (get it? ingredients float in soup)


# Now the following steps can be created and used together,
# thanks to their matching signatures:
@step(authors=["Friar Hugo"])
def make_soup(splits: List[tuple]) -> Soup:  # returns Soup
    return Soup(x for x, _ in splits)


@step(authors=["Abbot Mortimer"], input_info="a spoonful of Soup object")
def rate_soup(soup_sample: Soup) -> float:
    return 10 / 10


# further detail can also be added to metadata after step definition
rate_soup.contributors += ["Friar Hugo", "Sister Constance"]


pea_edibility_pipeline = Pipeline(
    title="Concept: a pipeline for soup",
    steps=(split_peas, make_soup, rate_soup),
    authors=pea_garden.authors,
)

# the complete pipeline is now also callable by itself
assert pea_edibility_pipeline([1, 2, 3]) == 10 / 10

pea_garden.pipelines += [pea_edibility_pipeline]

# writes complete metadata.json to current working directory
client.register_metadata(pea_garden)

# publishes metadata to search index
result = client.publish_garden(pea_garden, ["public"])

# propagates any authors of steps/pipelines as "contributors"
assert "Sister Constance" in pea_garden.contributors
