import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic.json import pydantic_encoder

from garden_ai.constants import GardenConstants
from garden_ai.gardens import Garden
from garden_ai.pipelines import RegisteredPipeline

LOCAL_STORAGE = Path(GardenConstants.GARDEN_DIR)
LOCAL_STORAGE.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger()


class LocalDataException(Exception):
    """Exception raised when a user's local data.json is corrupted"""

    pass


class PipelineNotFoundException(KeyError):
    """Exception raised when a Garden references an unknown pipeline DOI"""


class GardenNotFoundException(KeyError):
    """Exception raised when no Garden is found with a given DOI"""


class ResourceType(Enum):
    GARDEN = "gardens"
    PIPELINE = "pipelines"


resource_type_to_id_key = {
    ResourceType.GARDEN: "doi",
    ResourceType.PIPELINE: "doi",
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
    contents = json.dumps(data, default=pydantic_encoder)
    with open(LOCAL_STORAGE / "data.json", "w+") as f:
        f.write(contents)


def _put_notebook_base_image(notebook_path: Path, base_image: str) -> None:
    data = _read_local_db()
    nb_key = str(notebook_path.resolve())
    if "notebooks" not in data:
        data["notebooks"] = {}
    data["notebooks"][nb_key] = base_image
    _write_local_db(data)
    return


def _get_notebook_base_image(notebook_path: Path) -> Optional[str]:
    data = _read_local_db()
    nb_key = str(notebook_path.resolve())
    if "notebooks" not in data:
        return None
    return data["notebooks"].get(nb_key)


def _store_user_image_repo(repo: str) -> None:
    data = _read_local_db()
    data["user_image_repo"] = repo
    _write_local_db(data)


def _get_user_image_repo() -> Optional[str]:
    data = _read_local_db()
    maybe_repo = data.get("user_image_repo")
    return maybe_repo


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
    resource: Union[Garden, RegisteredPipeline],
    resource_type: ResourceType,
) -> None:
    resource_metadata = resource.dict()
    _put_resource_from_metadata(resource_metadata, resource_type)


def _make_obj_from_record(
    record: Dict, resource_type: ResourceType
) -> Union[Garden, RegisteredPipeline]:
    if resource_type is ResourceType.GARDEN:
        return Garden(**record)
    else:
        return RegisteredPipeline(**record)


def _get_resource_by_id(
    id_: str, resource_type: ResourceType
) -> Optional[Union[Garden, RegisteredPipeline]]:
    data = _read_local_db()
    resources = data.get(resource_type.value, {})
    if resources and id_ in resources:
        return _make_obj_from_record(resources[id_], resource_type)
    else:
        return None


def _get_resource_by_type(
    resource_type: ResourceType,
) -> Optional[List[Union[Garden, RegisteredPipeline]]]:
    data = _read_local_db()
    resource_data = data.get(resource_type.value, {})
    resource_objs = []
    if resource_data:
        for key, val in resource_data.items():
            resource_objs.append(_make_obj_from_record(val, resource_type))
        return resource_objs
    else:
        return None


def put_local_garden(garden: Garden):
    """Helper: write a record to 'local database' for a given Garden
    Overwrites any existing entry with the same doi in ~/.garden/data.json.

    Parameters
    ----------
    garden Garden
        The object to json-serialize and write/update in the local database.
        a TypeError will be raised if not a Garden.
    """
    _put_resource_from_obj(garden, resource_type=ResourceType.GARDEN)


def put_local_pipeline(pipeline: RegisteredPipeline):
    """Helper: write a record to 'local database' for a given Pipeline
    Overwrites any existing entry with the same doi in ~/.garden/data.json.

    Parameters
    ----------
    pipeline Pipeline
        The object to json-serialize and write/update in the local database.
        a TypeError will be raised if not a Pipeline.
    """
    _put_resource_from_obj(pipeline, resource_type=ResourceType.PIPELINE)


def get_local_garden_by_doi(doi: str) -> Optional[Garden]:
    """Helper: fetch a Garden record from ~/.garden/data.json.

    Parameters
    ----------
    doi str
        The DOI of the Garden you are fetching.

    Returns
    -------
    Optional[Garden]
    """
    return _get_resource_by_id(doi, ResourceType.GARDEN)  # type: ignore


def get_local_pipeline_by_doi(doi: str) -> Optional[RegisteredPipeline]:
    """Helper: fetch a Pipeline record from ~/.garden/data.json.

    Parameters
    ----------
    doi str
        The DOI of the Pipeline you are fetching.

    Returns
    -------
    Optional[RegisteredPipeline]
    """
    return _get_resource_by_id(doi, ResourceType.PIPELINE)  # type: ignore


def get_all_local_gardens() -> Optional[List[Garden]]:
    """Helper: fetch all Garden records from ~/.garden/data.json.

    Parameters
    ----------

    Returns
    -------
    Optional[Garden]
        If successful, a list of all the Garden objects in data.json.
    """
    return _get_resource_by_type(ResourceType.GARDEN)  # type: ignore


def get_all_local_pipelines() -> Optional[List[RegisteredPipeline]]:
    """Helper: fetch all pipeline records from ~/.garden/data.json.

    Parameters
    ----------

    Returns
    -------
    Optional[RegisteredPipeline]
        If successful, a list of all the RegisteredPipeline objects in data.json.
    """
    return _get_resource_by_type(ResourceType.PIPELINE)  # type: ignore
