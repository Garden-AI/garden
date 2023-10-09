import json

from globus_sdk import GlobusAPIError, SearchClient
from pydantic import ValidationError
from rich.traceback import install

from garden_ai.constants import GardenConstants
from garden_ai.gardens import PublishedGarden

install()


class RemoteGardenException(Exception):
    """Exception raised when a requested Garden cannot be found or published"""

    pass


def get_remote_garden_by_doi(doi: str, search_client: SearchClient) -> PublishedGarden:
    index_uuid = GardenConstants.GARDEN_INDEX_UUID
    try:
        res = search_client.get_subject(index_uuid, doi)
    except GlobusAPIError as e:
        if e.http_status == 404:
            raise RemoteGardenException(f"Could not find Garden with DOI {doi}") from e
        else:
            raise RemoteGardenException(
                f"Could not reach Globus Search Index {index_uuid}"
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
    index_uuid = GardenConstants.GARDEN_INDEX_UUID
    res = search_client.search(index_uuid, query, advanced=True)
    return res.text
