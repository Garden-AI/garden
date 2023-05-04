import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union
from uuid import UUID

from garden_ai.gardens import Garden
from garden_ai.pipelines import RegisteredPipeline
from garden_ai.mlmodel import RegisteredModel
from garden_ai.utils.misc import garden_json_encoder

LOCAL_STORAGE = Path("~/.garden").expanduser()
LOCAL_STORAGE.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger()


class LocalDataException(Exception):
    """Exception raised when a user's local data.json is corrupted"""

    pass


class PipelineNotFoundException(KeyError):
    """Exception raised when a Garden references an unknown pipeline uuid"""


class GardenNotFoundException(KeyError):
    """Exception raised when no Garden is found with a given uuid"""


class ResourceType(Enum):
    GARDEN = "gardens"
    PIPELINE = "pipelines"
    MODEL = "models"


resource_type_to_id_key = {
    ResourceType.GARDEN: "uuid",
    ResourceType.PIPELINE: "uuid",
    ResourceType.MODEL: "model_uri",
}


def _read_local_db() -> Dict:
    data = {}
    if (LOCAL_STORAGE / "data.json").exists():
        with open(LOCAL_STORAGE / "data.json", "r+") as f:
            raw_data = f.read()
            if raw_data:
                try:
                    data = json.loads(raw_data)
                except json.JSONDecodeError as e:
                    raise LocalDataException(
                        "Could not parse data.json as valid json"
                    ) from e
    return data


def _write_local_db(data: Dict) -> None:
    contents = json.dumps(data, default=garden_json_encoder)
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
    id_key = resource_type_to_id_key[resource_type]
    resources[str(resource_metadata[id_key])] = resource_metadata
    data[resource_type.value] = resources
    _write_local_db(data)


def _put_resource_from_obj(
    resource: Union[Garden, RegisteredPipeline, RegisteredModel],
    resource_type: ResourceType,
) -> None:
    resource_metadata = resource.dict()
    _put_resource_from_metadata(resource_metadata, resource_type)


def _make_obj_from_record(
    record: Dict, resource_type: ResourceType
) -> Union[Garden, RegisteredPipeline, RegisteredModel]:
    if resource_type is ResourceType.GARDEN:
        return Garden(**record)
    elif resource_type is ResourceType.PIPELINE:
        return RegisteredPipeline(**record)
    else:
        return RegisteredModel(**record)


def _get_resource_by_id(
    id_: Union[UUID, str], resource_type: ResourceType
) -> Optional[Union[Garden, RegisteredPipeline, RegisteredModel]]:
    data = _read_local_db()
    id_ = str(id_)
    resources = data.get(resource_type.value, {})
    if resources and id_ in resources:
        return _make_obj_from_record(resources[id_], resource_type)
    else:
        return None


def _get_resource_by_doi(
    doi: str, resource_type: ResourceType
) -> Optional[Union[Garden, RegisteredPipeline, RegisteredModel]]:
    data = _read_local_db()
    resources_by_uuid = data.get(resource_type.value, {})
    resources_by_doi = _reindex_by_doi(resources_by_uuid)
    if resources_by_doi and doi in resources_by_doi:
        return _make_obj_from_record(resources_by_doi[doi], resource_type)
    else:
        return None


def _reindex_by_doi(resources: dict) -> Dict:
    by_doi = {}
    for resource in resources.values():
        if "doi" in resource:
            by_doi[resource["doi"]] = resource
    return by_doi


def _delete_old_model_versions(model_name: str):
    data = _read_local_db()
    models_by_uri = data.get(ResourceType.MODEL.value, {})
    # Use `list` so that we don't delete items while iterating.
    for k, v in list(models_by_uri.items()):
        if v["model_name"] == model_name:
            del models_by_uri[k]
    data[ResourceType.MODEL.value] = models_by_uri
    _write_local_db(data)


def put_local_garden(garden: Garden):
    """Helper: write a record to 'local database' for a given Garden
    Overwrites any existing entry with the same uuid in ~/.garden/data.json.

    Parameters
    ----------
    garden Garden
        The object to json-serialize and write/update in the local database.
        a TypeError will be raised if not a Garden.
    """
    _put_resource_from_obj(garden, resource_type=ResourceType.GARDEN)


def put_local_pipeline(pipeline: RegisteredPipeline):
    """Helper: write a record to 'local database' for a given Pipeline
    Overwrites any existing entry with the same uuid in ~/.garden/data.json.

    Parameters
    ----------
    pipeline Pipeline
        The object to json-serialize and write/update in the local database.
        a TypeError will be raised if not a Pipeline.
    """
    _put_resource_from_obj(pipeline, resource_type=ResourceType.PIPELINE)


def get_local_garden_by_uuid(uuid: Union[UUID, str]) -> Optional[Garden]:
    """Helper: fetch a Garden record from ~/.garden/data.json.

    Parameters
    ----------
    uuid Union[UUID, str]
        The uuid of the Garden you are fetching.

    Returns
    -------
    Optional[Garden]
        If successful, a dictionary in the form given by Garden.json().
    """
    return _get_resource_by_id(uuid, ResourceType.GARDEN)  # type: ignore


def get_local_pipeline_by_uuid(uuid: Union[UUID, str]) -> Optional[RegisteredPipeline]:
    """Helper: fetch a Pipeline record from ~/.garden/data.json.

    Parameters
    ----------
    uuid Union[UUID, str]
        The uuid of the Pipeline you are fetching.

    Returns
    -------
    Optional[RegisteredPipeline]
    """
    return _get_resource_by_id(uuid, ResourceType.PIPELINE)  # type: ignore


def get_local_garden_by_doi(doi: str) -> Optional[Garden]:
    """Helper: fetch a Garden record from ~/.garden/data.json.

    Parameters
    ----------
    doi str
        The doi of the Garden you are fetching.

    Returns
    -------
    Optional[Garden]
    """
    return _get_resource_by_doi(doi, ResourceType.GARDEN)  # type: ignore


def get_local_pipeline_by_doi(doi: str) -> Optional[RegisteredPipeline]:
    """Helper: fetch a Pipeline record from ~/.garden/data.json.

    Parameters
    ----------
    doi str
        The doi of the Pipeline you are fetching.

    Returns
    -------
    Optional[RegisteredPipeline]
    """
    return _get_resource_by_doi(doi, ResourceType.PIPELINE)  # type: ignore


def put_local_model(model: RegisteredModel):
    """Helper: write a record to 'local database' for a given RegisteredModel
    Overwrites any existing entry with the same model_name in ~/.garden/data.json.

    Parameters
    ----------
    model Model
        The object to json-serialize and write/update in the local database.
        a TypeError will be raised if not a Model.
    """
    _delete_old_model_versions(model.model_name)
    _put_resource_from_obj(model, resource_type=ResourceType.MODEL)


def get_local_model_by_uri(model_uri: str):
    return _get_resource_by_id(model_uri, ResourceType.MODEL)  # type: ignore
