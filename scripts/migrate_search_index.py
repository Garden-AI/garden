import json
import os
from datetime import datetime
from typing import Annotated, List, TypeVar
from urllib.parse import quote
from uuid import UUID

from pydantic import AfterValidator, BaseModel, Field, HttpUrl, PlainSerializer
from pydantic_core import PydanticCustomError

from garden_ai import GardenClient, PublishedGarden
from garden_ai.constants import GardenConstants

CLIENT = GardenClient()
HEADERS = {"Authorization": CLIENT.garden_authorizer.get_authorization_header()}


# =============== PYDANTIC SCHEMAS COPIED FROM GARDEN BACKEND ================

T = TypeVar("T")


def _validate_unique_list(v: list[T]) -> list[T]:
    if len(v) != len(set(v)):
        raise PydanticCustomError("unique_list", "List must be unique")
    return v


UniqueList = Annotated[
    List[T],
    AfterValidator(_validate_unique_list),
    Field(json_schema_extra={"uniqueItems": True}, default_factory=list),
]
Url = Annotated[HttpUrl, PlainSerializer(lambda url: str(url), return_type=type(""))]


class _RepositoryMetadata(BaseModel):
    repo_name: str
    url: Url
    contributors: UniqueList[str] = Field(default_factory=list)


class _PaperMetadata(BaseModel):
    title: str
    authors: UniqueList[str] = Field(default_factory=list)
    doi: str | None
    citation: str | None


class _DatasetMetadata(BaseModel):
    title: str = Field(...)
    doi: str | None
    url: Url
    data_type: str | None
    repository: str


# protected_namespaces=() to allow model_* attribute names
class _ModelMetadata(BaseModel, protected_namespaces=()):
    model_identifier: str
    model_repository: str
    model_version: str | None


class EntrypointMetadata(BaseModel):
    doi: str
    doi_is_draft: bool
    title: str
    description: str | None
    year: str
    func_uuid: UUID
    container_uuid: UUID
    base_image_uri: str
    full_image_uri: str
    notebook_url: Url

    short_name: str
    function_text: str

    authors: UniqueList[str] = Field(default_factory=list)
    tags: UniqueList[str] = Field(default_factory=list)
    test_functions: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)

    models: list[_ModelMetadata] = Field(default_factory=list)
    repositories: list[_RepositoryMetadata] = Field(default_factory=list)
    papers: list[_PaperMetadata] = Field(default_factory=list)
    datasets: list[_DatasetMetadata] = Field(default_factory=list)


class EntrypointCreateRequest(EntrypointMetadata):
    owner_identity_id: UUID | None = None


class EntrypointMetadataResponse(EntrypointMetadata):
    owner_identity_id: UUID
    id: int


class GardenMetadata(BaseModel):
    title: str
    authors: UniqueList[str] = Field(default_factory=list)
    contributors: UniqueList[str] = Field(default_factory=list)
    doi: str
    doi_is_draft: bool | None = None
    description: str | None
    publisher: str = "Garden-AI"
    year: str = Field(default_factory=lambda: str(datetime.now().year))
    language: str = "en"
    tags: UniqueList[str] = Field(default_factory=list)
    version: str = "0.0.1"
    entrypoint_aliases: dict[str, str] = Field(default_factory=dict)


class GardenCreateRequest(GardenMetadata):
    entrypoint_ids: UniqueList[str] = Field(default_factory=list)
    owner_identity_id: UUID | None = None


class GardenMetadataResponse(GardenMetadata):
    id: int
    owner_identity_id: UUID | None
    entrypoints: list[EntrypointMetadataResponse] = Field(default_factory=list)


# ============== END COPIED SCHEMAS =================


def scrape_search_index(client: GardenClient, save_json=True) -> list[PublishedGarden]:
    """Collect every `PublishedGarden` whose contents are currently visible on the search index.

    Optionally, saves a backup copy of the search index contents locally.
    """
    index = GardenConstants.GARDEN_INDEX_UUID
    years = range(2020, 2025)
    query = " OR ".join([f"(year: {year})" for year in years])
    result = client.search_client.search(
        index_id=index, q=query, limit=100, advanced=True
    ).text
    data = json.loads(result)
    if save_json:
        ts = datetime.now().timestamp()
        env = os.environ["GARDEN_ENV"]  # fails unless prod is explicitly targeted
        filename = f"search-index-contents-{env}-{ts}.json"
        print(f"saving copy of search index to {filename}")
        with open(filename, "w+") as f:
            json.dump(data, f)

    published_gardens = [
        PublishedGarden(**data["gmeta"][i]["entries"][0]["content"])
        for i in range(len(data["gmeta"]))
    ]
    return published_gardens


def build_request_bodies(
    search_index_gardens: list[PublishedGarden], scramble_dois=False
) -> tuple[list[EntrypointCreateRequest], list[GardenCreateRequest]]:
    """Convert list of `PublishedGarden`s, containing nested entrypoints, into
    two normalized lists of the respective PUT request payloads.
    """
    entrypoint_payloads = []
    garden_payloads = []

    for garden in search_index_gardens:
        garden_data = garden.model_dump()
        # normalized garden data only tracks DOIs
        garden_data["entrypoint_ids"] = []

        for entrypoint in garden.entrypoints:
            # extract function_text before dropping steps field
            if entrypoint.function_text is None:
                if entrypoint.steps:
                    entrypoint.function_text = entrypoint.steps[0].function_text
                else:
                    print(
                        f"skipping entrypoint with no steps or function text: {entrypoint}"
                    )
                    continue
            # handle missing values for migrated dlhub models
            if not entrypoint.base_image_uri:
                entrypoint.base_image_uri = "n/a - dlhub"
            if not entrypoint.full_image_uri:
                entrypoint.full_image_uri = "n/a - dlhub"
            if not entrypoint.notebook_url:
                entrypoint.notebook_url = "https://www.dlhub.org"

            # add doi to garden payload
            garden_data["entrypoint_ids"] += [entrypoint.doi]
            # create the request payload
            entrypoint_data = entrypoint.model_dump(exclude={"steps"})
            entrypoint_payloads += [EntrypointCreateRequest(**entrypoint_data)]

        del garden_data["entrypoints"]
        garden_payloads += [GardenCreateRequest(**garden_data)]

    return entrypoint_payloads, garden_payloads


def put_entrypoint(client: GardenClient, entrypoint: EntrypointCreateRequest) -> dict:
    url = f"/entrypoints/{quote(entrypoint.doi, safe='')}"
    payload = entrypoint.model_dump(mode="json")
    return client.backend_client._put(url, payload)


def put_garden(client: GardenClient, garden: GardenCreateRequest):
    url = f"/gardens/{quote(garden.doi, safe='')}"
    payload = garden.model_dump(mode="json")
    return client.backend_client._put(url, payload)


if __name__ == "__main__":
    client = GardenClient()
    print("scraping search index")
    published_gardens = scrape_search_index(client, save_json=True)

    print(f"collected {len(published_gardens)} gardens from search index")
    entrypoints, gardens = build_request_bodies(published_gardens)

    print(f"creating/updating {len(entrypoints)} entrypoints")
    for entrypoint in entrypoints:
        put_entrypoint(client, entrypoint)

    print(f"creating/updating {len(gardens)} gardens")
    for garden in gardens:
        put_garden(client, garden)
