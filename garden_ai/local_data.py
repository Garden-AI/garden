_IS_DISABLED = True


import json  # noqa: E402
import logging  # noqa: E402
from enum import Enum  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Union  # noqa: E402

from pydantic_core import to_jsonable_python  # noqa: E402

from garden_ai.constants import GardenConstants  # noqa: E402
from garden_ai.entrypoints import RegisteredEntrypoint  # noqa: E402
from garden_ai.gardens import Garden  # noqa: E402

LOCAL_STORAGE = Path(GardenConstants.GARDEN_DIR)
LOCAL_STORAGE.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger()


class LocalDataException(Exception):
    """Exception raised when a user's local data.json is corrupted"""

    pass


class EntrypointNotFoundException(KeyError):
    """Exception raised when a Garden references an unknown entrypoint DOI"""


class GardenNotFoundException(KeyError):
    """Exception raised when no Garden is found with a given DOI"""


class ResourceType(Enum):
    GARDEN = "gardens"
    ENTRYPOINT = "entrypoints"


resource_type_to_id_key = {
    ResourceType.GARDEN: "doi",
    ResourceType.ENTRYPOINT: "doi",
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
    contents = json.dumps(data, default=to_jsonable_python)
    with open(LOCAL_STORAGE / "data.json", "w+") as f:
        f.write(contents)


def _store_user_email(email: str) -> None:
    data = _read_local_db()
    data["user_email"] = email
    _write_local_db(data)


def _clear_user_email() -> None:
    data = _read_local_db()
    if "user_email" in data:
        del data["user_email"]
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
    resource: Union[Garden, RegisteredEntrypoint],
    resource_type: ResourceType,
) -> None:
    resource_metadata = resource.model_dump()
    _put_resource_from_metadata(resource_metadata, resource_type)


def _make_obj_from_record(
    record: Dict, resource_type: ResourceType
) -> Union[Garden, RegisteredEntrypoint]:
    if resource_type is ResourceType.GARDEN:
        return Garden(**record)
    else:
        return RegisteredEntrypoint(**record)


def _get_resource_by_id(
    id_: str, resource_type: ResourceType
) -> Optional[Union[Garden, RegisteredEntrypoint]]:
    data = _read_local_db()
    resources = data.get(resource_type.value, {})
    if resources and id_ in resources:
        return _make_obj_from_record(resources[id_], resource_type)
    else:
        return None


def _get_resource_by_type(
    resource_type: ResourceType,
) -> Optional[List[Union[Garden, RegisteredEntrypoint]]]:
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


def put_local_entrypoint(entrypoint: RegisteredEntrypoint):
    """Helper: write a record to 'local database' for a given Entrypoint
    Overwrites any existing entry with the same doi in ~/.garden/data.json.

    Parameters
    ----------
    entrypoint Entrypoint
        The object to json-serialize and write/update in the local database.
        a TypeError will be raised if not an Entrypoint.
    """
    _put_resource_from_obj(entrypoint, resource_type=ResourceType.ENTRYPOINT)


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


def delete_local_garden_by_doi(doi: str) -> None:
    """Helper: delete a Garden record from ~/.garden/data.json.

    Parameters
    ----------
    doi str
        The DOI of the Garden you are deleting.
    """
    data = _read_local_db()
    if "gardens" in data and doi in data["gardens"]:
        del data["gardens"][doi]
        _write_local_db(data)


def get_local_entrypoint_by_doi(doi: str) -> Optional[RegisteredEntrypoint]:
    """Helper: fetch an Entrypoint record from ~/.garden/data.json.

    Parameters
    ----------
    doi str
        The DOI of the Entrypoint you are fetching.

    Returns
    -------
    Optional[RegisteredEntrypoint]
    """
    return _get_resource_by_id(doi, ResourceType.ENTRYPOINT)  # type: ignore


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


def get_all_local_entrypoints() -> Optional[List[RegisteredEntrypoint]]:
    """Helper: fetch all entrypoint records from ~/.garden/data.json.

    Parameters
    ----------

    Returns
    -------
    Optional[RegisteredEntrypoint]
        If successful, a list of all the RegisteredEntrypoint objects in data.json.
    """
    return _get_resource_by_type(ResourceType.ENTRYPOINT)  # type: ignore
