import json
import logging
import os
import time
from pathlib import Path
from typing import List, Union

from rich import print
from rich.prompt import Prompt
import typer

import requests
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

from garden_ai.gardens import Garden
from garden_ai.pipelines import Pipeline

# garden-dev index
GARDEN_INDEX_UUID = "58e4df29-4492-4e7d-9317-b27eba62a911"

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

        self.authorizer = self._authenticate()
        self.searchauthorizer = self._create_search_authorizer()
        self.search_client = (
            SearchClient(authorizer=self.searchauthorizer)
            if not search_client
            else search_client
        )
        self.garden_authorizer = self._create_garden_authorizer()

    def _do_login_flow(self):
        self.auth_client.oauth2_start_flow(
            requested_scopes=[
                GroupsClient.scopes.view_my_groups_and_memberships,
                SearchClient.scopes.ingest,
                GardenClient.scopes.action_all,  # "https://auth.globus.org/scopes/0948a6b0-a622-4078-b0a4-bfd6d77d65cf/action_all"
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

    def _create_search_authorizer(self):
        if not self.auth_key_store.file_exists():
            # do a login flow, getting back initial tokens
            response = self._do_login_flow()

            if not response:
                raise AuthException

            # now store the tokens and pull out the Groups tokens
            self.auth_key_store.store(response)
            tokens = response.by_resource_server[SearchClient.resource_server]
        else:
            # otherwise, we already did login; load the tokens from that file
            tokens = self.auth_key_store.get_token_data(SearchClient.resource_server)
        # construct the RefreshTokenAuthorizer which writes back to storage on refresh
        authorizer = RefreshTokenAuthorizer(
            tokens["refresh_token"],
            self.auth_client,
            access_token=tokens["access_token"],
            expires_at=tokens["expires_at_seconds"],
            on_refresh=self.auth_key_store.on_refresh,
        )
        return authorizer

    def _create_garden_authorizer(self):
        if not self.auth_key_store.file_exists():
            # do a login flow, getting back initial tokens
            response = self._do_login_flow()

            if not response:
                raise AuthException

            # now store the tokens and pull out the Groups tokens
            self.auth_key_store.store(response)

            tokens = response.by_resource_server[GardenClient.scopes.resource_server]
        else:
            # otherwise, we already did login; load the tokens from that file
            tokens = self.auth_key_store.get_token_data(
                GardenClient.scopes.resource_server
            )

        # construct the RefreshTokenAuthorizer which writes back to storage on refresh
        authorizer = RefreshTokenAuthorizer(
            tokens["refresh_token"],
            self.auth_client,
            access_token=tokens["access_token"],
            expires_at=tokens["expires_at_seconds"],
            on_refresh=self.auth_key_store.on_refresh,
        )
        return authorizer

    def _authenticate(self):
        if not self.auth_key_store.file_exists():
            # do a login flow, getting back initial tokens
            response = self._do_login_flow()

            if not response:
                raise AuthException

            # now store the tokens and pull out the Groups tokens
            self.auth_key_store.store(response)
            tokens = response.by_resource_server[GroupsClient.resource_server]
        else:
            # otherwise, we already did login; load the tokens from that file
            tokens = self.auth_key_store.get_token_data(GroupsClient.resource_server)

        # construct the RefreshTokenAuthorizer which writes back to storage on refresh
        authorizer = RefreshTokenAuthorizer(
            tokens["refresh_token"],
            self.auth_client,
            access_token=tokens["access_token"],
            expires_at=tokens["expires_at_seconds"],
            on_refresh=self.auth_key_store.on_refresh,
        )
        return authorizer

    def create_garden(self, authors: List[str] = [], title: str = "", **kwargs) -> Garden:
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
        endpoint = os.environ.get(
            "GARDEN_ENDPOINT",
            "https://nu3cetwc84.execute-api.us-east-1.amazonaws.com/garden_prod",
        )
        try:
            url = f"{endpoint}/doi"
        except KeyError:
            logger.error(
                "Expected environment variable GARDEN_ENDPOINT to be set. No DOI has been generated."
            )
            raise

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
        """Make a `Garden` object's metadata (and any pipelines' metadata) discoverable; mint DOIs via DataCite.

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
            with open(out_dir / f"{garden.garden_id}.json", "w+") as f:
                f.write(garden.json())

    def publish_garden(self, garden=None, visibility="Public"):
        # Takes a garden_id UUID as a subject, and a garden_doc dict, and
        # publishes to the GARDEN_INDEX_UUID index.  Polls to discover status,
        # and returns the Task document:
        # https://docs.globus.org/api/search/reference/get_task/#task

        # Sanity check visibility--if it is a string, make it a list of strings
        if isinstance(visibility, str):
            visibility = [visibility]

        gmeta_ingest = {
            "subject": str(garden.garden_id),
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
