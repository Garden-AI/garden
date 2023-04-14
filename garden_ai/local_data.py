import json
import logging
from enum import Enum
from typing import Dict, Union, Optional
from pathlib import Path
from uuid import UUID

from garden_ai.gardens import Garden
from garden_ai.pipelines import Pipeline

LOCAL_STORAGE = Path("~/.garden").expanduser()
LOCAL_STORAGE.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger()


class ResourceType(Enum):
    GARDEN = "gardens"
    PIPELINE = "pipelines"


# This guy needs a try/except
def _read_local_db() -> Dict:
    data = {}
    if (LOCAL_STORAGE / "data.json").exists():
        with open(LOCAL_STORAGE / "data.json", "r+") as f:
            raw_data = f.read()
            if raw_data:
                data = json.loads(raw_data)
    return data


def _write_local_db(data: Dict) -> None:
    contents = json.dumps(data)
    with open(LOCAL_STORAGE / "data.json", "w+") as f:
        f.write(contents)


def _store_user_email(email: str) -> None:
    data = _read_local_db()
    data["user_email"] = email
    _write_local_db(data)


def _get_user_email() -> str:
    data = _read_local_db()
    maybe_email = data.get("user_email")
    return str(maybe_email) if maybe_email else "unknown"


def _put_resource_from_metadata(
    resource_metadata: Dict, resource_type: ResourceType
) -> None:
    data = _read_local_db()
    resources = data.get(resource_type.value, {})
    resources[str(resource_metadata["uuid"])] = resource_metadata
    data[resource_type.value] = resources
    _write_local_db(data)


def _put_resource_from_pydantic(resource: Union[Garden, Pipeline]) -> None:
    resource_type = (
        ResourceType.GARDEN if isinstance(resource, Garden) else ResourceType.PIPELINE
    )
    resource_metadata = json.loads(resource.json())
    _put_resource_from_metadata(resource_metadata, resource_type)


def _get_resource_by_uuid(
    uuid: Union[UUID, str], resource_type: ResourceType
) -> Optional[Dict]:
    data = _read_local_db()
    uuid = str(uuid)
    resources = data.get(resource_type.value, {})
    if resources and uuid in resources:
        return resources[uuid]
    else:
        return None


def _get_resource_by_doi(doi: str, resource_type: ResourceType) -> Optional[Dict]:
    data = _read_local_db()
    resources_by_uuid = data.get(resource_type.value, {})
    resources_by_doi = _reindex_by_doi(resources_by_uuid)
    if resources_by_doi and doi in resources_by_doi:
        return resources_by_doi[doi]
    else:
        return None


def _reindex_by_doi(resources: dict) -> Dict:
    by_doi = {}
    for resource in resources.values():
        if "doi" in resource:
            by_doi[resource["doi"]] = resource
    return by_doi


def put_local_garden(garden: Garden):
    _put_resource_from_pydantic(garden)


def put_local_garden_from_metadata(garden_metadata: Dict):
    _put_resource_from_metadata(garden_metadata, ResourceType.GARDEN)


def put_local_pipeline(pipeline: Pipeline):
    _put_resource_from_pydantic(pipeline)


def get_local_garden_by_uuid(uuid: Union[UUID, str]) -> Optional[Dict]:
    return _get_resource_by_uuid(uuid, ResourceType.GARDEN)


def get_local_pipeline_by_uuid(uuid: Union[UUID, str]) -> Optional[Dict]:
    return _get_resource_by_uuid(uuid, ResourceType.PIPELINE)


def get_local_garden_by_doi(doi: str) -> Optional[Dict]:
    return _get_resource_by_doi(doi, ResourceType.GARDEN)


def get_local_pipeline_by_doi(doi: str) -> Optional[Dict]:
    return _get_resource_by_doi(doi, ResourceType.PIPELINE)
