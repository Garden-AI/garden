from .client import GardenClient
from .constants import GardenConstants
from .entrypoints import Entrypoint, entrypoint, entrypoint_test
from .gardens import Garden
from .model_connectors.model_utils import create_connector
from .schemas.entrypoint import (
    DatasetMetadata,
    EntrypointMetadata,
    PaperMetadata,
    RepositoryMetadata,
)

garden_entrypoint = entrypoint  # backwards compatibility with old decorator name

__all__ = [
    "GardenConstants",
    "GardenClient",
    "Garden",
    "EntrypointMetadata",
    "DatasetMetadata",
    "PaperMetadata",
    "RepositoryMetadata",
    "Entrypoint",
    "garden_entrypoint",
    "entrypoint",
    "entrypoint_test",
    "create_connector",
    "get_garden",
]


def get_garden(doi: str) -> Garden:
    """Get a Garden by its DOI

    If not already authed, this will trigger the auth flow and
    prompt the caller for a Globus auth token.

    Args:
        doi: str The DOI of the Garden you are looking for

    Example:
        ```python
        import garden_ai

        doi = "10.26311/6phn-gv02"
        garden = garden_ai.get_garden(doi)
        inputs = [<your data here>]
        result = garden.some_function(inputs)
        ```
    """
    gc = GardenClient()
    return gc.get_garden(doi)
