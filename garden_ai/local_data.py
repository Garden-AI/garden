import json
import logging
from typing import Dict, List, Union, Optional
from pathlib import Path
from uuid import UUID

from garden_ai.gardens import Garden
from garden_ai.pipelines import Pipeline
from garden_ai.utils.misc import JSON

LOCAL_STORAGE = Path("~/.garden").expanduser()
LOCAL_STORAGE.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger()


def _read_local_db() -> Dict:
    """Helper: load JSON contents of local storage and return as a dict."""
    data = {}
    # read existing entries into memory, if any
    if (LOCAL_STORAGE / "data.json").exists():
        with open(LOCAL_STORAGE / "data.json", "r+") as f:
            raw_data = f.read()
            if raw_data:
                data = json.loads(raw_data)
    return data


def _write_local_db(data: Dict) -> None:
    """Helper: JSON-serialize and write ``contents`` to ~/.garden/data.json."""
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


def put_local_garden_metadata(garden_metadata: Dict) -> None:
    """Helper: write a record to 'local database' for a given Garden

    Overwrites any existing entry with the same uuid in ~/.garden/data.json.

    Parameters
    ----------
    garden_metadata : Dict
        Garden metadata in the format given by Garden.json()

    """
    _put_garden(garden_metadata)


def put_local_garden(garden: Garden) -> None:
    """Helper: write a record to 'local database' for a given Garden

    Overwrites any existing entry with the same uuid in ~/.garden/data.json.

    Parameters
    ----------
    garden : Garden
        The object to json-serialize and write/update in the local database.
        a TypeError will be raised if not a Garden.

    """
    if not isinstance(garden, Garden):
        raise TypeError(f"Expected Garden object, got: {type(garden)}.")
    _put_garden(json.loads(garden.json()))


def _put_garden(garden_metadata: Dict):
    data = _read_local_db()
    key, val = str(garden_metadata['uuid']), garden_metadata
    local_gardens = data.get("gardens", {})
    local_gardens[key] = val
    data["gardens"] = local_gardens

    _write_local_db(data)


def get_local_garden(uuid: Union[UUID, str]) -> Optional[JSON]:
    """Helper: fetch a record from 'local database'

    Find entry with key matching ``uuid`` and return the associated metadata
    extracted from ``~/.garden/db/data.json``

    Parameters
    ----------
    uuid : UUID
        The uuid corresponding to the desired Garden or Pipeline.

    Returns
    -------
    Optional[JSON]
        If successful, the JSON string corresponding to the metadata of the
        object with the given uuid.
    """
    data = _read_local_db()

    uuid = str(uuid)
    local_gardens = data.get("gardens", {})
    if local_gardens and uuid in local_gardens:
        return json.dumps(local_gardens[uuid])
    else:
        logger.error(f"No garden found locally with uuid: {uuid}.")
        return None


def put_local_pipeline(pipeline: Pipeline) -> None:
    """Helper: write a record to 'local database' for a given Pipeline.

    Overwrites any existing entry with the same ``uuid``.

    Parameters
    ----------
    pipeline : Pipeline
        The object to json-serialize and write/update in the local database.
        a TypeError will be raised if not a Pipeline.

    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected pipeline object, got: {type(pipeline)}.")
    data = {}
    # read existing entries into memory, if any
    if (LOCAL_STORAGE / "data.json").exists():
        with open(LOCAL_STORAGE / "data.json", "r+") as f:
            raw_data = f.read()
            if raw_data:
                data = json.loads(raw_data)

    # update data['pipelines'], leaving data['gardens'] etc unmodified
    pipelines = data.get("pipelines", {})
    key, val = str(pipeline.uuid), pipeline.json()
    pipelines[key] = json.loads(val)
    data["pipelines"] = pipelines
    contents = json.dumps(data)

    with open(LOCAL_STORAGE / "data.json", "w+") as f:
        f.write(contents)


def get_local_pipeline(uuid: Union[UUID, str]) -> Optional[JSON]:
    """Helper: fetch a pipeline record from 'local database', if one exists.

    Find entry with key matching ``uuid`` and return the associated metadata
    extracted from ``~/.garden/db/data.json``

    Parameters
    ----------
    uuid : UUID
        The uuid corresponding to the desired Pipeline.

    Returns
    -------
    Optional[JSON]
        If successful, the JSON string corresponding to the metadata of the
        object with the given uuid.
    """
    uuid = str(uuid)
    with open(LOCAL_STORAGE / "data.json", "r+") as f:
        raw_contents = f.read()
        if raw_contents:
            data: Dict[str, Dict] = json.loads(raw_contents)
        else:
            logger.error("Local storage is empty; could not find by uuid.")
            return None

    if "pipelines" in data and uuid in data["pipelines"]:
        result = data["pipelines"][uuid]
        return json.dumps(result)
    else:
        logger.error(f"No pipeline found locally with uuid: {uuid}.")
        return None
