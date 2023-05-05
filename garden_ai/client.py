# mypy: disable-error-code="import"
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Union
from uuid import UUID

import requests
import typer
from globus_compute_sdk import Client
from globus_sdk import (
    AuthAPIError,
    AuthClient,
    GroupsClient,
    NativeAppAuthClient,
    RefreshTokenAuthorizer,
    SearchClient,
    GlobusHTTPResponse,
)
from globus_sdk.scopes import AuthScopes, ScopeBuilder, SearchScopes
from globus_sdk.tokenstorage import SimpleJSONFileAdapter
from mlflow.tracking.request_header.registry import _request_header_provider_registry
from rich import print
from rich.prompt import Prompt

import garden_ai.funcx_bandaid.serialization_patch  # type: ignore # noqa: F401
from garden_ai import local_data
from garden_ai.gardens import Garden
from garden_ai.globus_compute.containers import build_container
from garden_ai.globus_compute.login_manager import ComputeLoginManager
from garden_ai.globus_compute.remote_functions import register_pipeline
from garden_ai.globus_search import garden_search

from garden_ai.local_data import GardenNotFoundException, PipelineNotFoundException
from garden_ai.mlflow_bandaid.binary_header_provider import (
    BinaryContentTypeHeaderProvider,
)
from garden_ai.mlmodel import LocalModel, RegisteredModel, upload_to_model_registry
from garden_ai.pipelines import Pipeline, RegisteredPipeline
from garden_ai.utils.misc import extract_email_from_globus_jwt

GARDEN_ENDPOINT = os.environ.get(
    "GARDEN_ENDPOINT",
    "https://nu3cetwc84.execute-api.us-east-1.amazonaws.com/garden_prod",
)

COMPUTE_RESOURCE_SERVER_NAME = "funcx_service"

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
        self.openid_authorizer = self._create_authorizer(
            AuthClient.scopes.resource_server
        )
        self.groups_authorizer = self._create_authorizer(
            GroupsClient.scopes.resource_server
        )
        self.search_authorizer = self._create_authorizer(
            SearchClient.scopes.resource_server
        )
        self.compute_authorizer = self._create_authorizer(COMPUTE_RESOURCE_SERVER_NAME)
        self.search_client = (
            SearchClient(authorizer=self.search_authorizer)
            if not search_client
            else search_client
        )
        self.garden_authorizer = self._create_authorizer(
            GardenClient.scopes.resource_server
        )

        self.compute_client = self._make_compute_client()
        self._set_up_mlflow_env()

    def _set_up_mlflow_env(self):
        os.environ["MLFLOW_TRACKING_TOKEN"] = self.garden_authorizer.access_token
        os.environ["MLFLOW_TRACKING_URI"] = GARDEN_ENDPOINT + "/mlflow"
        _request_header_provider_registry.register(BinaryContentTypeHeaderProvider)

    def _make_compute_client(self):
        scope_to_authorizer = {
            AuthScopes.openid: self.openid_authorizer,
            SearchScopes.all: self.search_authorizer,
            Client.FUNCX_SCOPE: self.compute_authorizer,
        }
        compute_login_manager = ComputeLoginManager(scope_to_authorizer)
        return Client(login_manager=compute_login_manager, do_version_check=False)

    def _do_login_flow(self):
        self.auth_client.oauth2_start_flow(
            requested_scopes=[
                AuthClient.scopes.openid,
                AuthClient.scopes.email,
                GroupsClient.scopes.view_my_groups_and_memberships,
                SearchClient.scopes.all,
                GardenClient.scopes.action_all,
                Client.FUNCX_SCOPE,
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
            local_data._store_user_email(email)
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
        """Initialize and return a pipeline object.

        If this pipeline's UUID has been used before to register a function for
        remote execution, reuse the (funcx/globus compute) ID for consistency.

        NOTE: this means that local modifications to a pipeline will not be
        reflected when executing remotely until the pipeline is re-registered.
        """
        data = dict(kwargs)
        if authors:
            data["authors"] = authors
        if title:
            data["title"] = title

        pipeline = Pipeline(**data)
        record = local_data.get_local_pipeline_by_uuid(pipeline.uuid)
        if record:
            logger.info("Found pre-registered pipeline. Reusing DOI.")
            pipeline.doi = record.doi

        return pipeline

    def register_model(self, local_model: LocalModel) -> RegisteredModel:
        registered_model = upload_to_model_registry(local_model)
        local_data.put_local_model(registered_model)
        return registered_model

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

        def get_existing_doi() -> Optional[str]:
            # check for existing doi, either on object or in db
            registered_obj: Optional[Union[Garden, RegisteredPipeline]]
            if isinstance(obj, Garden):
                registered_obj = local_data.get_local_garden_by_uuid(obj.uuid)
            else:
                registered_obj = local_data.get_local_pipeline_by_uuid(obj.uuid)

            return registered_obj.doi if registered_obj else None

        existing_doi = obj.doi or get_existing_doi()

        if existing_doi and not force:
            logger.info(
                "existing DOI found, not requesting new DOI. Pass `force=true` to override this behavior."
            )
            return existing_doi

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

    def build_container(self, pipeline: Pipeline) -> str:
        built_container_uuid = build_container(self.compute_client, pipeline)
        return built_container_uuid

    def register_pipeline(self, pipeline: Pipeline, container_uuid: str) -> str:
        func_uuid = register_pipeline(self.compute_client, pipeline, container_uuid)
        pipeline.func_uuid = UUID(func_uuid)
        pipeline.doi = self._mint_doi(pipeline)
        registered = RegisteredPipeline.from_pipeline(pipeline)
        local_data.put_local_pipeline(registered)
        return func_uuid

    def get_registered_pipeline(
        self, identifier: Union[UUID, str]
    ) -> RegisteredPipeline:
        """Return a callable ``RegisteredPipeline`` corresponding to the given uuid/doi.

        Parameters
        ----------
        identifier : Union[UUID, str]
            The previously registered pipeline's DOI or UUID. Raises an
            exception if not found.

        Returns
        -------
        RegisteredPipeline
            Instance of ``RegisteredPipeline``, which can be run on
            a specified remote Globus Compute endpoint (and knows how to
            set the appropriate MLFlow environment variables).

        Raises
        ------
        PipelineNotFoundException
            Raised when no known pipeline exists with the given identifier.
        """
        is_doi = "/" in str(identifier)
        if is_doi:
            registered = local_data.get_local_pipeline_by_doi(str(identifier))
        else:
            registered = local_data.get_local_pipeline_by_uuid(identifier)

        if registered is None:
            raise PipelineNotFoundException(
                f"Could not find any pipelines with identifier {identifier}."
            )

        self._set_up_mlflow_env()
        registered._env_vars = {
            "MLFLOW_TRACKING_TOKEN": os.environ["MLFLOW_TRACKING_TOKEN"],
            "MLFLOW_TRACKING_URI": os.environ["MLFLOW_TRACKING_URI"],
        }

        return registered

    def _fresh_mlflow_vars(self):
        self._set_up_mlflow_env()
        return {
            "MLFLOW_TRACKING_TOKEN": os.environ["MLFLOW_TRACKING_TOKEN"],
            "MLFLOW_TRACKING_URI": os.environ["MLFLOW_TRACKING_URI"],
        }

    def get_email(self) -> str:
        return local_data._get_user_email()

    def get_local_garden(self, identifier: Union[UUID, str]) -> Garden:
        """Return a registered ``Garden`` corresponding to the given uuid/doi.

        Any ``RegisteredPipelines`` registered to the Garden will be callable
        as attributes on the garden by their (registered) short_name, e.g.
            ```python
                my_garden = client.get_local_garden('garden-doi-or-uuid')
                #  pipeline would have been registered with short_name='my_pipeline'
                my_garden.my_pipeline(*args, endpoint='where-to-execute')
            ```
        Tip: To access the pipeline by a different name, use ``my_garden.rename_pipeline(old_name, new_name)``.
        To persist a new name for a pipeline, re-register it to the garden and specify an alias.

        Parameters
        ----------
        identifier : Union[UUID, str]
            The previously registered Garden's DOI or UUID. Raises an
            exception if not found.

        """
        is_doi = "/" in str(identifier)
        if is_doi:
            registered = local_data.get_local_garden_by_doi(str(identifier))
        else:
            registered = local_data.get_local_garden_by_uuid(identifier)

        if registered is None:
            raise GardenNotFoundException(
                f"Could not find any Gardens with identifier {identifier}."
            )
        self._set_up_mlflow_env()
        registered._env_vars = {
            "MLFLOW_TRACKING_TOKEN": os.environ["MLFLOW_TRACKING_TOKEN"],
            "MLFLOW_TRACKING_URI": os.environ["MLFLOW_TRACKING_URI"],
        }
        return registered

    def publish_garden_metadata(self, garden: Garden) -> GlobusHTTPResponse:
        """
        Takes a garden, and publishes to the GARDEN_INDEX_UUID index.  Polls
        to discover status, and returns the Task document.
        Task document has a task["state"] field that can be FAILED or SUCCESS.

        Returns
        -------
        https://docs.globus.org/api/search/reference/get_task/#task
        """
        return garden_search.publish_garden_metadata(garden, self.search_client)

    def search(self, query: str) -> str:
        """
        Given a Globus Search advanced query, returns JSON Globus Search result string with gardens as entries.
        """
        return garden_search.search_gardens(query, self.search_client)

    def get_published_garden(self, identifier: str) -> Garden:
        """
        Queries Globus Search for the garden with this DOI.

        Parameters
        ----------
        identifier: The doi or uuid of the garden you want.

        Returns
        -------
        Garden populated with metadata from remote metadata record.

        """
        is_doi = "/" in str(identifier)
        if is_doi:
            return garden_search.get_remote_garden_by_doi(
                identifier, self._fresh_mlflow_vars(), self.search_client
            )
        else:
            return garden_search.get_remote_garden_by_uuid(
                identifier, self._fresh_mlflow_vars(), self.search_client
            )
