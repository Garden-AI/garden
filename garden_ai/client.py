import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Union, Optional
from uuid import UUID

import requests
import typer
from globus_sdk import (
    AuthAPIError,
    AuthClient,
    GroupsClient,
    NativeAppAuthClient,
    RefreshTokenAuthorizer,
    SearchClient,
)
from globus_sdk.scopes import ScopeBuilder
from globus_sdk.tokenstorage import SimpleJSONFileAdapter
from pydantic import ValidationError
from rich import print
from rich.prompt import Prompt

from garden_ai.gardens import Garden
from garden_ai.pipelines import Pipeline
from garden_ai.utils import JSON, extract_email_from_globus_jwt
from garden_ai.mlmodel import upload_model

# garden-dev index
GARDEN_INDEX_UUID = "58e4df29-4492-4e7d-9317-b27eba62a911"
GARDEN_ENDPOINT = os.environ.get(
    "GARDEN_ENDPOINT",
    "https://nu3cetwc84.execute-api.us-east-1.amazonaws.com/garden_prod",
)

LOCAL_STORAGE = Path("~/.garden").expanduser()
LOCAL_STORAGE.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger()


class AuthException(Exception):
    pass


GardenScopes = ScopeBuilder(
    "0948a6b0-a622-4078-b0a4-bfd6d77d65cf", known_url_scopes=["action_all"]
)


class GardenClient:
    """
    Main class for interacting with the Garden service
    Holds helper operations for performing common tasks
    with the Garden service.

    Will authenticate with GlobusAuth, storing generated keys in the users .garden
    directory

    Raises:
         AuthException: if the user cannot authenticate
    """

    scopes = GardenScopes

    def __init__(
        self, auth_client: AuthClient = None, search_client: SearchClient = None
    ):
        key_store_path = Path(os.path.expanduser("~/.garden"))
        key_store_path.mkdir(exist_ok=True)
        self.auth_key_store = SimpleJSONFileAdapter(
            os.path.join(key_store_path, "tokens.json")
        )
        self.client_id = os.environ.get(
            "GARDEN_CLIENT_ID", "cf9f8938-fb72-439c-a70b-85addf1b8539"
        )

        self.auth_client = (
            NativeAppAuthClient(self.client_id) if not auth_client else auth_client
        )

        self.groups_authorizer = self._create_authorizer(
            GroupsClient.scopes.resource_server
        )
        self.search_authorizer = self._create_authorizer(
            SearchClient.scopes.resource_server
        )
        self.search_client = (
            SearchClient(authorizer=self.search_authorizer)
            if not search_client
            else search_client
        )
        self.garden_authorizer = self._create_authorizer(
            GardenClient.scopes.resource_server
        )

        self._set_up_mlflow_env()

    def _set_up_mlflow_env(self):
        os.environ["MLFLOW_TRACKING_TOKEN"] = self.garden_authorizer.access_token
        os.environ["MLFLOW_TRACKING_URI"] = GARDEN_ENDPOINT + "/mlflow"

    def _do_login_flow(self):
        self.auth_client.oauth2_start_flow(
            requested_scopes=[
                AuthClient.scopes.openid,
                AuthClient.scopes.email,
                GroupsClient.scopes.view_my_groups_and_memberships,
                SearchClient.scopes.ingest,
                GardenClient.scopes.action_all,
            ],
            refresh_tokens=True,
        )
        authorize_url = self.auth_client.oauth2_get_authorize_url()

        print(
            f"Authenticating with Globus in your default web browser: \n\n{authorize_url}"
        )
        time.sleep(2)
        typer.launch(authorize_url)

        auth_code = Prompt.ask("Please enter the code here ").strip()

        try:
            tokens = self.auth_client.oauth2_exchange_code_for_tokens(auth_code)
            return tokens
        except AuthAPIError:
            logger.fatal("Invalid Globus auth token received. Exiting")
            return None

    def _create_authorizer(self, resource_server: str):
        if not self.auth_key_store.file_exists():
            # do a login flow, getting back initial tokens
            response = self._do_login_flow()

            if not response:
                raise AuthException

            # now store the tokens and pull out the Groups tokens
            self.auth_key_store.store(response)
            tokens = response.by_resource_server[resource_server]

            email = extract_email_from_globus_jwt(response.data["id_token"])
            self._store_user_email(email)
        else:
            # otherwise, we already did login; load the tokens from that file
            tokens = self.auth_key_store.get_token_data(resource_server)
        # construct the RefreshTokenAuthorizer which writes back to storage on refresh
        authorizer = RefreshTokenAuthorizer(
            tokens["refresh_token"],
            self.auth_client,
            access_token=tokens["access_token"],
            expires_at=tokens["expires_at_seconds"],
            on_refresh=self.auth_key_store.on_refresh,
        )
        return authorizer

    def create_garden(
        self, authors: List[str] = [], title: str = "", **kwargs
    ) -> Garden:
        """Construct a new Garden object, optionally populating any number of metadata fields from `kwargs`.

        Up to user preference, metadata (e.g. `title="My Garden"` or
        `version="1.0.0"`) can be provided here as kwargs.

        This might be useful if, for example, one wanted to build a Garden starting
        from an already-existing Garden or pre-populated dict of template
        metadata. Otherwise, the user is free to incrementally populate or
        replace even the Garden object's required fields (e.g. `pea_garden.title
        = "Experiments on Plant Hybridization"`) at any time -- field validation
        is still performed.

        Parameters
        ----------
        authors : List[str]
            The personal names of main researchers/authors involved in
            cultivating the Garden. Names should be formatted "Family, Given",
            e.g. `authors=["Mendel, Gregor"]`. Affiliations/institutional
            relationships should be added via the Garden object helper method
            `add_affiliation`, e.g.  `pea_garden.add_affiliation({"Mendel,
            Gregor": "St Thomas' Abbey"})`. (NOTE: add_affiliation not yet implemented)

        title : str
            An official name or title for the Garden. This attribute must be set
            in order to register a DOI.

        **kwargs :
            Metadata for the new Garden object. Keyword arguments matching
            required or recommended fields will be (where necessary) coerced to the
            appropriate type and validated per the Garden metadata spec.

        Examples
        --------
            gc = GardenClient()
            pea_garden = gc.create_garden(
                authors=["Mendel, Gregor"],
                title="Experiments on Plant Hybridization",
                subjects=["Peas"]
            )
            pea_garden.year = 1863
            pea_garden.subjects += ["Genetics"] # (didn't have the word for it earlier)
        """
        data = dict(kwargs)
        if authors:
            data["authors"] = authors
        if title:
            data["title"] = title
        return Garden(**data)

    def create_pipeline(
        self, authors: Optional[List[str]] = None, title: Optional[str] = None, **kwargs
    ) -> Pipeline:
        data = dict(kwargs)
        if authors:
            data["authors"] = authors
        if title:
            data["title"] = title
        return Pipeline(**data)

    def log_model(
        self, model_path: str, model_name: str, extra_pip_requirements: List[str] = None
    ) -> str:
        email = self._get_user_email()
        full_model_name = upload_model(
            model_path, model_name, email, extra_pip_requirements=extra_pip_requirements
        )
        return full_model_name

    def _mint_doi(
        self, obj: Union[Garden, Pipeline], force: bool = False, test: bool = True
    ) -> str:
        """Register a new "findable" doi with DataCite via Garden backend.

        Expects environment variable GARDEN_ENDPOINT to be set (not including `/doi`).

        Parameters
        ----------
        obj : Union[Garden, Pipeline]
            the Pipeline or Garden object wanting a new DOI.
        force : bool
            Mint a new DOI even if one exists (note that old ones stay
            "findable" forever - see
            https://support.datacite.org/docs/best-practices-for-datacite-members)
        test : bool
            toggle which garden backend endpoint to hit; we do not yet have a
            test endpoint so test=True raises NotImplementedError.

        Raises
        ------
        NotImplementedError
            see `test`

        """
        if not test:
            raise NotImplementedError
        elif obj.doi and not force:
            logger.info(
                "existing DOI found, not requesting new DOI. Pass `force=true` to override this behavior."
            )
            return obj.doi

        logger.info("Requesting DOI")
        url = f"{GARDEN_ENDPOINT}/doi"

        header = {
            "Content-Type": "application/vnd.api+json",
            "Authorization": self.garden_authorizer.get_authorization_header(),
        }
        metadata = json.loads(obj.datacite_json())
        metadata.update(event="publish", url="https://thegardens.ai")
        payload = {"data": {"type": "dois", "attributes": metadata}}
        r = requests.post(
            url,
            headers=header,
            json=payload,
        )
        try:
            r.raise_for_status()
            doi = r.json()["doi"]
        except requests.HTTPError:
            logger.error(f"{r.text}")
            raise
        else:
            return doi

    def register_metadata(self, garden: Garden, out_dir=None):
        """
        NOTE: this is mostly vestigial until we're at the point of implementing
        ``$ garden-ai {garden, pipeline, model(?)} register`` and know what should be
        here instead.

        Make a `Garden` object's metadata (and any pipelines' metadata)
        discoverable; mint DOIs via DataCite.

        This will perform validation on the metadata fields and (if successful)
        write all of the Garden's (including its pipelines) metadata to a
        `"metadata.json"` file.

        Parameters
        ----------
        garden : Garden
            A Garden object with user's ready-to-publish metadata. See `Garden`
            docstring for explanation of fields.

        out_dir : Union[PathLike, str]
            Directory in which a copy of the Garden's metadata is written on
            successful registration. Defaults to current working directory.

        Raises
        ------
        ValidationError

        """
        out_dir = Path(out_dir) if out_dir else Path.cwd()
        try:
            for p in garden.pipelines:
                p.doi = self._mint_doi(p)
            garden.doi = self._mint_doi(garden)
            garden.validate()
        except ValidationError as e:
            logger.error(e)
            raise
        else:
            with open(out_dir / f"{garden.uuid}.json", "w+") as f:
                f.write(garden.json())

    def _read_local_db(self) -> Dict:
        """Helper: load JSON contents of local storage and return as a dict."""
        data = {}
        # read existing entries into memory, if any
        if (LOCAL_STORAGE / "data.json").exists():
            with open(LOCAL_STORAGE / "data.json", "r+") as f:
                raw_data = f.read()
                if raw_data:
                    data = json.loads(raw_data)
        return data

    def _write_local_db(self, data: Dict) -> None:
        """Helper: JSON-serialize and write ``contents`` to ~/.garden/data.json."""
        contents = json.dumps(data)
        with open(LOCAL_STORAGE / "data.json", "w+") as f:
            f.write(contents)
        return

    def _store_user_email(self, email: str) -> None:
        data = self._read_local_db()
        data["user_email"] = email
        self._write_local_db(data)

    def _get_user_email(self) -> str:
        data = self._read_local_db()
        maybe_email = data.get("user_email")
        return str(maybe_email) if maybe_email else "unknown"

    def put_local_garden(self, garden: Garden) -> None:
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
        data = self._read_local_db()

        key, val = str(garden.uuid), json.loads(garden.json())
        local_gardens = data.get("gardens", {})
        local_gardens[key] = val
        data["gardens"] = local_gardens

        self._write_local_db(data)
        return

    def get_local_garden(self, uuid: Union[UUID, str]) -> Optional[JSON]:
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
        data = self._read_local_db()

        uuid = str(uuid)
        local_gardens = data.get("gardens", {})
        if local_gardens and uuid in local_gardens:
            return json.dumps(local_gardens[uuid])
        else:
            logger.error(f"No garden found locally with uuid: {uuid}.")
            return None

    def put_local_pipeline(self, pipeline: Pipeline) -> None:
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
        return

    def get_local_pipeline(self, uuid: Union[UUID, str]) -> Optional[JSON]:
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

    def publish_garden(self, garden=None, visibility="Public"):
        # Takes a garden_id UUID as a subject, and a garden_doc dict, and
        # publishes to the GARDEN_INDEX_UUID index.  Polls to discover status,
        # and returns the Task document:
        # https://docs.globus.org/api/search/reference/get_task/#task

        # Sanity check visibility--if it is a string, make it a list of strings
        if isinstance(visibility, str):
            visibility = [visibility]

        gmeta_ingest = {
            "subject": str(garden.uuid),
            "visible_to": visibility,
            "content": json.loads(garden.json()),
        }

        publish_result = self.search_client.create_entry(
            GARDEN_INDEX_UUID, gmeta_ingest
        )

        task_result = self.search_client.get_task(publish_result["task_id"])
        while not task_result["state"] in {"FAILED", "SUCCESS"}:
            time.sleep(5)
            task_result = self.search_client.get_task(publish_result["task_id"])
        return task_result
