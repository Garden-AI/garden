import base64
import json
import logging
import os
import time
import urllib
from pathlib import Path
from typing import Callable, Optional, Union
from uuid import UUID

import mixpanel  # type: ignore
import typer
from globus_compute_sdk import Client
from globus_compute_sdk.sdk.login_manager import ComputeScopes
from globus_compute_sdk.sdk.login_manager.tokenstore import get_token_storage_adapter
from globus_compute_sdk.serialize.concretes import DillCodeTextInspect
from globus_sdk import (
    AccessTokenAuthorizer,
    AuthAPIError,
    AuthLoginClient,
    ClientCredentialsAuthorizer,
    ConfidentialAppAuthClient,
    GroupsClient,
    NativeAppAuthClient,
    RefreshTokenAuthorizer,
)
from globus_sdk.authorizers import GlobusAuthorizer
from globus_sdk.scopes import ScopeBuilder
from globus_sdk.tokenstorage import SimpleJSONFileAdapter
from rich import print
from rich.prompt import Prompt

from garden_ai.backend_client import BackendClient
from garden_ai.constants import GardenConstants
from garden_ai.entrypoints import Entrypoint
from garden_ai.garden_file_adapter import GardenFileAdapter
from garden_ai.gardens import Garden
from garden_ai.schemas.entrypoint import RegisteredEntrypointMetadata
from garden_ai.schemas.garden import GardenMetadata
from garden_ai.hpc_gardens.alpha_fold import AlphaFoldGarden
from garden_ai.utils._meta import make_function_to_register
from modal.cli._traceback import setup_rich_traceback

logger = logging.getLogger()
# modal helper replacement for rich.traceback.install
setup_rich_traceback()


class AuthException(Exception):
    pass


GardenScopes = ScopeBuilder(
    "0948a6b0-a622-4078-b0a4-bfd6d77d65cf",
    known_url_scopes=["action_all", "test_scope"],
)


class GardenClient:
    """
    Main class for interacting with the Garden service.

    Provides public `get_garden` and `get_entrypoint` methods, and handles authentication with Globus Auth.
    """

    scopes = GardenScopes
    _mixpanel_track = None

    def __init__(
        self,
        auth_client: Optional[Union[AuthLoginClient, ConfidentialAppAuthClient]] = None,
    ):
        key_store_path = Path(GardenConstants.GARDEN_DIR)
        key_store_path.mkdir(exist_ok=True)
        self.garden_key_store = SimpleJSONFileAdapter(GardenConstants.GARDEN_KEY_STORE)
        self.compute_key_store = get_token_storage_adapter()
        self.auth_key_store = GardenFileAdapter(
            self.garden_key_store, self.compute_key_store
        )

        self.client_id = os.environ.get(
            "GARDEN_CLIENT_ID", "cf9f8938-fb72-439c-a70b-85addf1b8539"
        )

        # If auth_client is type AuthLoginClient or None, do an
        # Authorization Code Grant and make RefreshTokenAuthorizers.
        # If auth_client is type ConfidentialAppAuthClient, do a
        # Client Credentials Grant and make ClientCredentialsAuthorizers
        if (
            isinstance(auth_client, AuthLoginClient)
            and not isinstance(auth_client, ConfidentialAppAuthClient)
        ) or not auth_client:
            self.auth_client = (
                NativeAppAuthClient(self.client_id) if not auth_client else auth_client
            )
            self.openid_authorizer = self._create_authorizer(
                AuthLoginClient.scopes.resource_server
            )
            self.groups_authorizer = self._create_authorizer(
                GroupsClient.scopes.resource_server
            )
            self.compute_authorizer = self._create_authorizer(
                ComputeScopes.resource_server
            )
            self.garden_authorizer = self._create_authorizer(
                GardenClient.scopes.resource_server
            )
        elif isinstance(auth_client, ConfidentialAppAuthClient):
            self.auth_client = auth_client
            self.openid_authorizer = ClientCredentialsAuthorizer(
                self.auth_client,
                f"{AuthLoginClient.scopes.openid} {AuthLoginClient.scopes.email}",
            )
            self.groups_authorizer = ClientCredentialsAuthorizer(
                self.auth_client, GroupsClient.scopes.view_my_groups_and_memberships
            )
            self.compute_authorizer = ClientCredentialsAuthorizer(
                self.auth_client, Client.FUNCX_SCOPE
            )
            self.garden_authorizer = ClientCredentialsAuthorizer(
                self.auth_client,
                GardenClient.scopes.test_scope,
            )

        self.compute_client = self._make_compute_client()
        self.backend_client = BackendClient(self.garden_authorizer)
        # get_email_and_user_identity_id call ensures user info is present on the backend,
        # which means user is member of the demo endpoint group
        email, user_id = self.get_email_and_user_identity_id()
        logger.info(f"logged in user: {email}")

        if os.environ.get("GARDEN_ENV") in ("dev", "local"):
            self._mixpanel_track = None
        else:
            self._mixpanel_track = self._make_mixpanel_track_fn(user_id, email)

    def _get_garden_access_token(self):
        self.garden_authorizer.ensure_valid_token()
        return self.garden_authorizer.access_token

    def _make_compute_client(self):
        return Client(
            do_version_check=False,
            code_serialization_strategy=DillCodeTextInspect(),
        )

    def _make_mixpanel_track_fn(self, user_id, user_email) -> Optional[Callable]:
        try:
            mp = mixpanel.Mixpanel(
                GardenConstants.MIXPANEL_TOKEN,
                consumer=mixpanel.Consumer(retry_limit=1),
            )
            mp.people_set(user_id, {"$email": user_email})
        except Exception as e:
            # Swallow the error and keep going - don't let mixpanel errors crash the app
            logger.debug(f"Failed to initialize Mixpanel with error: {e}")
            return None

        def _track(event_name, event_properties):
            try:
                mp.track(user_id, event_name, event_properties)
            except Exception as e:
                # Swallow the error and keep going - don't let mixpanel errors crash the app
                logger.debug(
                    f"Failed to track event {event_name} with properties {event_properties} due to error: {e}"
                )

        return _track

    def _display_notebook_link(self, link):
        from IPython.display import HTML, display

        display(
            HTML(
                f'<a href="{link}" rel="noopener noreferrer" target="_blank">{link}</a>'
            )
        )

    def _do_login_flow(self):
        self.auth_client.oauth2_start_flow(
            requested_scopes=[
                AuthLoginClient.scopes.openid,
                AuthLoginClient.scopes.email,
                AuthLoginClient.scopes.manage_projects,
                GroupsClient.scopes.all,
                GardenClient.scopes.test_scope,
                GardenClient.scopes.action_all,
                ComputeScopes.all,
            ],
            refresh_tokens=True,
        )
        authorize_url = self.auth_client.oauth2_get_authorize_url()

        try:
            __IPYTHON__  # Check if running in notebook. '__IPYTHON__' is defined if in one.
            print("Authenticating with Globus in your default web browser: \n\n")
            self._display_notebook_link(
                authorize_url
            )  # Must display url as html to render properly in notebooks
        except NameError:
            print(
                f"Authenticating with Globus in your default web browser: \n\n{authorize_url}"
            )

        time.sleep(2)
        if not os.environ.get("GARDEN_DISABLE_BROWSER"):
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

        else:
            # otherwise, we already did login; load the tokens from that file
            tokens = self.auth_key_store.get_token_data(resource_server)
        # construct the RefreshTokenAuthorizer which writes back to storage on refresh
        try:
            authorizer: GlobusAuthorizer = RefreshTokenAuthorizer(
                tokens["refresh_token"],
                self.auth_client,
                access_token=tokens["access_token"],
                expires_at=tokens["expires_at_seconds"],
                on_refresh=self.auth_key_store.on_refresh,
            )
        except TypeError:
            # If there is no refresh token, try using an access token
            authorizer = AccessTokenAuthorizer(tokens["access_token"])
        return authorizer

    def register_entrypoint_doi(self, entrypoint_doi: str) -> None:
        """
        Makes an entrypoint's DOI registered and findable with DataCite via the Garden backend.
        """
        entrypoint_meta = self.backend_client.get_entrypoint_metadata(entrypoint_doi)
        self._update_datacite(entrypoint_meta, register_doi=True)
        entrypoint_meta.doi_is_draft = False
        self.backend_client.put_entrypoint_metadata(entrypoint_meta)

    def _mint_draft_doi(self) -> str:
        """Register a new draft DOI with DataCite via Garden backend."""

        logger.info("Requesting draft DOI")
        payload = {
            "data": {"type": "dois", "attributes": {}}
        }  # required data is filled in on the backend
        return self.backend_client.mint_doi_on_datacite(payload)

    def _update_datacite(
        self,
        obj: GardenMetadata | RegisteredEntrypointMetadata,
        register_doi: bool = False,
    ) -> None:
        logger.info("Requesting update to DOI")
        metadata = json.loads(obj._datacite_json())

        # "publish" in the event field moves the DOI from draft state to findable state
        # https://support.datacite.org/docs/how-do-i-make-a-findable-doi-with-the-rest-api
        if register_doi:
            doi = urllib.parse.quote(obj.doi, safe="")
            metadata.update(
                event="publish",
                url=f"https://thegardens.ai/#/garden/{doi}",
            )

        payload = {"data": {"type": "dois", "attributes": metadata}}
        self.backend_client.update_doi_on_datacite(payload)
        logger.info("Update request succeeded")

    def get_entrypoint(self, doi: str) -> Entrypoint:
        """Return the callable Entrypoint associated with the given DOI.

        Parameters:
            doi: The entrypoint's DOI. Raises an exception if not found.

        Returns:
            The callable Entrypoint object. Can execute remotely on a specified Globus Compute endpoint.

        """
        return self.backend_client.get_entrypoint(doi)

    def get_email(self) -> str:
        user_email, _ = self.get_email_and_user_identity_id()
        return user_email

    def get_user_identity_id(self) -> str:
        _, user_id = self.get_email_and_user_identity_id()
        return user_id

    def get_email_and_user_identity_id(self) -> tuple[str, str]:
        user_data = self.backend_client.get_user_info()
        user_email = user_data.get("email") or user_data.get("username")
        assert user_email is not None, "Failed to find user email"
        user_identity_id = user_data["identity_id"]
        assert user_identity_id is not None, "Failed to find user identity"
        return user_email, user_identity_id

    def upload_notebook(self, notebook_contents: dict, notebook_name: str) -> str:
        """
        POSTs a notebook's contents to the backend /notebook route
        so that we can store a link to the full notebook contents in the entrypoint metadata.
        """
        username = self.get_email()
        try:
            return self.backend_client.upload_notebook(
                notebook_contents, username, notebook_name
            )
        except Exception as e:
            raise Exception(
                f"Request to Garden backend to publish notebook failed with error: {str(e)}"
            )

    def get_garden(self, doi: str) -> Garden:
        """
        Return the published Garden associated with the given DOI.
        Parameters:
            doi: The DOI of the garden. Raises an exception if not found.
        Returns:
            The published Garden object. Any [entrypoints][garden_ai.Entrypoint] in the garden can be called like methods on this object.
        """
        if doi.lower() == "alphafold2":
            return AlphaFoldGarden(client=self, doi=doi)

        garden = self.backend_client.get_garden(doi)
        return garden

    def _get_auth_config_for_ecr_push(self) -> dict:
        """
        Calls the Garden backend to get a short-lived boto3 session with ECR permissions.
        Uses that session to get an authorization token from AWS for ECR Public.
        Uses the authorization token to construct an auth_config dict we can pass to the Docker client.
        """
        session = self.backend_client.get_docker_push_session()
        ecr_client = session.client("ecr-public")
        response = ecr_client.get_authorization_token()
        auth_data = response["authorizationData"]
        if isinstance(auth_data, list):
            auth_data = auth_data[0]
        password = (
            base64.b64decode(auth_data["authorizationToken"]).decode().split(":")[1]
        )
        auth_config = {"username": "AWS", "password": password}
        return auth_config

    def _register_and_publish_from_user_image(
        self,
        base_image_uri: str,
        full_image_uri: str,
        notebook_url: str,
        metadata: dict,
    ):
        """Register entrypoints update affected gardens from a user's finished notebook session image.

        Parameters:
        - base_image_uri: str
            The public location of the base image that the user image was built on top of.
        - full_image_uri: str
            The public location of the full user image built by running the
            notebook. This is the image we register with globus compute.
        - notebook_url: str
            URL that points to the full contents of the notebook.
        - metadata: dict
            metadata for entrypoints defined in the full image (i.e. the
            contents of the metadata.json extracted from the image)
        """

        container_uuid = UUID(
            self.compute_client.register_container(full_image_uri, "docker")
        )

        for function_name, record in metadata.items():
            # register function & populate remaining metadata fields
            to_register = make_function_to_register(function_name)
            func_uuid = self.compute_client.register_function(
                to_register, container_uuid=str(container_uuid), public=True
            )
            doi = self._mint_draft_doi()
            registered_meta = RegisteredEntrypointMetadata(
                doi=doi,
                doi_is_draft=True,
                func_uuid=func_uuid,
                container_uuid=container_uuid,
                base_image_uri=base_image_uri,
                full_image_uri=full_image_uri,
                notebook_url=notebook_url,
                **record,
            )
            self._update_datacite(registered_meta)

            self.backend_client.put_entrypoint_metadata(registered_meta)
