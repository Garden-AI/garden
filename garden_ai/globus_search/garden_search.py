from globus_sdk import SearchClient

from garden_ai.constants import GardenConstants


def search_gardens(query: str, search_client: SearchClient) -> str:
    index_uuid = GardenConstants.GARDEN_INDEX_UUID
    res = search_client.search(index_uuid, query, advanced=True)
    return res.text
