import json

from garden_ai.gardens import PublishedGarden
from globus_sdk import SearchClient, GlobusAPIError
from pydantic import ValidationError
from rich.traceback import install

# garden-dev index
GARDEN_INDEX_UUID = "58e4df29-4492-4e7d-9317-b27eba62a911"

install()


class RemoteGardenException(Exception):
    """Exception raised when a requested Garden cannot be found or published"""

    pass


def get_remote_garden_by_doi(doi: str, search_client: SearchClient) -> PublishedGarden:
    try:
        res = search_client.get_subject(GARDEN_INDEX_UUID, doi)
    except GlobusAPIError as e:
        if e.http_status == 404:
            raise RemoteGardenException(f"Could not find Garden with DOI {doi}") from e
        else:
            raise RemoteGardenException(
                f"Could not reach Globus Search Index {GARDEN_INDEX_UUID}"
            ) from e
    try:
        garden_meta = json.loads(res.text)["entries"][0]["content"]
        garden = PublishedGarden(**garden_meta)
    except (ValueError, KeyError, IndexError, ValidationError) as e:
        raise RemoteGardenException(
            f"Could not parse search response {res.text}"
        ) from e
    return garden


def search_gardens(query: str, search_client: SearchClient) -> str:
    res = search_client.search(GARDEN_INDEX_UUID, query, advanced=True)
    return res.text
