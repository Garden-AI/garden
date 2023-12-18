# mypy: disable-error-code="import"
import json
import logging
import os
import time
import base64
from pathlib import Path
from typing import List, Union
from uuid import UUID

import docker
import typer
from globus_compute_sdk import Client
from globus_compute_sdk.sdk.login_manager.tokenstore import get_token_storage_adapter
from globus_compute_sdk.serialize.concretes import DillCodeTextInspect
from globus_sdk import (
    AuthAPIError,
    AuthClient,
    ClientCredentialsAuthorizer,
    ConfidentialAppAuthClient,
    GroupsClient,
    NativeAppAuthClient,
    RefreshTokenAuthorizer,
    SearchClient,
)
from globus_sdk.scopes import ScopeBuilder
from globus_sdk.tokenstorage import SimpleJSONFileAdapter
from rich import print
from rich.prompt import Prompt

from garden_ai import GardenConstants, local_data
from garden_ai.backend_client import BackendClient
from garden_ai.containers import extract_metadata_from_image
from garden_ai.garden_file_adapter import GardenFileAdapter
from garden_ai.gardens import Garden, PublishedGarden
from garden_ai.globus_search import garden_search
from garden_ai.local_data import GardenNotFoundException, EntrypointNotFoundException
from garden_ai.entrypoints import Paper, RegisteredEntrypoint, Repository
from garden_ai.utils._meta import make_function_to_register
from garden_ai.utils.misc import extract_email_from_globus_jwt

COMPUTE_RESOURCE_SERVER_NAME = "funcx_service"

logger = logging.getLogger()


class AuthException(Exception):
    pass


GardenScopes = ScopeBuilder(
    "0948a6b0-a622-4078-b0a4-bfd6d77d65cf",
    known_url_scopes=["action_all", "test_scope"],
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
        self,
        auth_client: Union[AuthClient, ConfidentialAppAuthClient] = None,
        search_client: SearchClient = None,
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

        # If auth_client is type AuthClient or None, do an
        # Authorization Code Grant and make RefreshTokenAuthorizers.
        # If auth_client is type ConfidentialAppAuthClient, do a
        # Client Credentials Grant and make ClientCredentialsAuthorizers
        if (
            isinstance(auth_client, AuthClient)
            and not isinstance(auth_client, ConfidentialAppAuthClient)
        ) or not auth_client:
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
            self.compute_authorizer = self._create_authorizer(
                COMPUTE_RESOURCE_SERVER_NAME
            )
            self.search_client = (
                SearchClient(authorizer=self.search_authorizer)
                if not search_client
                else search_client
            )
            self.garden_authorizer = self._create_authorizer(
                GardenClient.scopes.resource_server
            )
        elif isinstance(auth_client, ConfidentialAppAuthClient):
            self.auth_client = auth_client
            self.openid_authorizer = ClientCredentialsAuthorizer(
                self.auth_client,
                f"{AuthClient.scopes.openid} {AuthClient.scopes.email}",
            )
            self.groups_authorizer = ClientCredentialsAuthorizer(
                self.auth_client, GroupsClient.scopes.view_my_groups_and_memberships
            )
            self.search_authorizer = ClientCredentialsAuthorizer(
                self.auth_client, SearchClient.scopes.all
            )
            self.compute_authorizer = ClientCredentialsAuthorizer(
                self.auth_client, Client.FUNCX_SCOPE
            )
            self.search_client = SearchClient(authorizer=self.search_authorizer)
            self.garden_authorizer = ClientCredentialsAuthorizer(
                self.auth_client, GardenClient.scopes.test_scope
            )

            local_data._store_user_email(GardenConstants.GARDEN_TEST_EMAIL)

        self.compute_client = self._make_compute_client()
        self.backend_client = BackendClient(self.garden_authorizer)

    def _get_garden_access_token(self):
        self.garden_authorizer.ensure_valid_token()
        return self.garden_authorizer.access_token

    def _make_compute_client(self):
        return Client(
            do_version_check=False,
            code_serialization_strategy=DillCodeTextInspect(),
        )

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
                AuthClient.scopes.openid,
                AuthClient.scopes.email,
                GroupsClient.scopes.view_my_groups_and_memberships,
                SearchClient.scopes.all,
                GardenClient.scopes.test_scope,
                Client.FUNCX_SCOPE,
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
            pea_garden.tags += ["Genetics"] # (didn't have the word for it earlier)
        """
        data = dict(kwargs)
        if authors:
            data["authors"] = authors
        if title:
            data["title"] = title
        data["doi"] = data.get("doi") or self._mint_draft_doi()

        return Garden(**data)

    def _mint_draft_doi(self) -> str:
        """Register a new draft DOI with DataCite via Garden backend."""

        logger.info("Requesting draft DOI")
        payload = {
            "data": {"type": "dois", "attributes": {}}
        }  # required data is filled in on the backend
        return self.backend_client.mint_doi_on_datacite(payload)

    def _update_datacite(
        self, obj: Union[PublishedGarden, RegisteredEntrypoint]
    ) -> None:
        logger.info("Requesting update to DOI")
        metadata = json.loads(obj.datacite_json())
        metadata.update(event="publish", url=f"https://thegardens.ai/{obj.doi}")

        payload = {"data": {"type": "dois", "attributes": metadata}}
        self.backend_client.update_doi_on_datacite(payload)
        logger.info("Update request succeeded")

    def add_paper(self, title: str, doi: str, **kwargs) -> None:
        """Adds a ``Paper`` to a ``RegisteredEntrypoint`` corresponding to the given DOI.

        Parameters
        ----------
        doi : str
            The previously registered entrypoint's DOI. Raises an
            exception if not found.

        title : str
            An official name or title for the paper.

        **kwargs :
            Metadata for the new Paper object. Keyword arguments matching
            required or recommended fields will be (where necessary) coerced to the appropriate type.
            May include: List[str] authors and Optional[str] citation.

        Raises
        ------
        EntrypointNotFoundException
            Raised when no known entrypoint exists with the given identifier.
        """
        entrypoint = local_data.get_local_entrypoint_by_doi(doi)
        if not entrypoint:
            raise EntrypointNotFoundException(
                f"Could not find any entrypoints with DOI {doi}."
            )
        data = dict(kwargs)
        if title:
            data["title"] = title
        paper = Paper(**data)
        entrypoint.papers.append(paper)
        local_data.put_local_entrypoint(entrypoint)

    def add_repository(self, doi: str, url: str, repo_name: str, **kwargs) -> None:
        """Adds a ``Repository`` to a ``RegisteredEntrypoint`` corresponding to the given DOI.

        Parameters
        ----------
        doi : str
            The previously registered entrypoint's DOI. Raises an
            exception if not found.

        title : str
            An official name or title for the repository.

        url: str
            The url to access this repository.

        **kwargs :
            Metadata for the new Repository object. Keyword arguments matching
            required or recommended fields will be (where necessary) coerced to the appropriate type.
            May include: List[str] contributors.

        Raises
        ------
        EntrypointNotFoundException
            Raised when no known entrypoint exists with the given identifier.
        """
        data = dict(kwargs)
        if doi:
            data["doi"] = doi
        if url:
            data["url"] = url
        if repo_name:
            data["repo_name"] = repo_name
        entrypoint = local_data.get_local_entrypoint_by_doi(doi)
        if not entrypoint:
            raise EntrypointNotFoundException(
                f"Could not find any entrypoints with DOI {doi}."
            )

        repository = Repository(**data)
        entrypoint.repositories.append(repository)
        local_data.put_local_entrypoint(entrypoint)

    def get_registered_entrypoint(self, doi: str) -> RegisteredEntrypoint:
        """Return a callable ``RegisteredEntrypoint`` corresponding to the given DOI.

        Parameters
        ----------
        doi : str
            The previously registered entrypoint's DOI. Raises an
            exception if not found.

        Returns
        -------
        RegisteredEntrypoint
            Instance of ``RegisteredEntrypoint``, which can be run on
            a specified remote Globus Compute endpoint.

        Raises
        ------
        EntrypointNotFoundException
            Raised when no known entrypoint exists with the given identifier.
        """
        entrypoint = local_data.get_local_entrypoint_by_doi(doi)

        if entrypoint is None:
            raise EntrypointNotFoundException(
                f"Could not find any entrypoints with DOI {doi}."
            )
        return entrypoint

    def get_email(self) -> str:
        return local_data._get_user_email()

    def get_local_garden(self, doi: str) -> Garden:
        """Return a registered ``Garden`` corresponding to the given DOI.

        Any ``RegisteredEntrypoints`` registered to the Garden will be callable
        as attributes on the garden by their (registered) short_name, e.g.
            ```python
                my_garden = client.get_local_garden('garden-doi')
                #  entrypoint would have been registered with short_name='my_entrypoint'
                my_garden.my_entrypoint(*args, endpoint='where-to-execute')
            ```
        Tip: To access the entrypoint by a different name, use ``my_garden.rename_entrypoint(entrypoint_id, new_name)``.
        To persist a new name for an entrypoint, re-register it to the garden and specify an alias.

        Parameters
        ----------
        doi : str
            The previously registered Garden's DOI. Raises an
            exception if not found.

        """
        garden = local_data.get_local_garden_by_doi(doi)

        if garden is None:
            raise GardenNotFoundException(
                f"Could not find any Gardens with identifier {doi}."
            )
        return garden

    def publish_garden_metadata(self, garden: Garden) -> None:
        """
        Publishes a Garden's expanded_json to the backend /garden-search-route,
        making it visible on our Globus Search index.
        """
        published = PublishedGarden.from_garden(garden)

        self._update_datacite(published)
        try:
            self.backend_client.publish_garden_metadata(published)
        except Exception as e:
            raise Exception(
                f"Request to Garden backend to publish garden failed with error: {str(e)}"
            )

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

    def search(self, query: str) -> str:
        """
        Given a Globus Search advanced query, returns JSON Globus Search result string with gardens as entries.
        """
        return garden_search.search_gardens(query, self.search_client)

    def clone_published_garden(self, doi: str, *, silent: bool = False) -> Garden:
        """
        Queries Globus Search for the garden with the given DOI
        and creates a local clone of it that can be modified.

        NOTE: the clone will have a different DOI than the original

        Parameters
        ----------
        doi: The DOI of the garden you want to clone.
        silent: Whether or not to print any messages.

        Returns
        -------
        Garden populated with metadata from the remote metadata record.
        """
        published = self.get_published_garden(doi)

        for entrypoint in published.entrypoints:
            local_data.put_local_entrypoint(entrypoint)

        data = published.dict()
        del data["doi"]  # the clone should not retain the DOI

        garden = self.create_garden(**data)

        if not silent:
            log_msg = f"Garden {doi} successfully cloned locally and given replacement DOI {garden.doi}."
            logger.info(log_msg)
            print(log_msg)

        return garden

    def get_published_garden(self, doi: str) -> PublishedGarden:
        """
        Queries Globus Search for the garden with this DOI.

        Parameters
        ----------
        doi: The DOI of the garden you want.

        Returns
        -------
        PublishedGarden populated with metadata from the remote metadata record.

        """
        garden = garden_search.get_remote_garden_by_doi(doi, self.search_client)
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
        docker_client: docker.DockerClient,
        image: docker.models.images.Image,
        base_image_uri: str,
        full_image_uri: str,
        notebook_url: str,
    ):
        """Register entrypoints and (re-)publish affected gardens from a user's finished notebook session image.

        Parameters:
        - docker_client : docker.DockerClient
        - image : docker.models.images.Image
            The result of a successful
            `containers.build_notebook_session_image`; should have both the
            final `"session.pkl"` and "`metadata.json`" files generated by
            `scripts.save_session_and_metadata` during the build process.
        - base_image_uri: str
            The public location of the base image that the user image was built on top of.
        - full_image_uri: str
            The public location of the full user image built by running the notebook.
        - notebook_url: str
            URL that points to the full contents of the notebook.

        Raises:
        - ValueError
            When attempting to add an entrypoint to a garden which does not exist
            in local data.
        """
        dirty_gardens = set()  # it's good for gardens to get dirty, actually

        container_uuid = UUID(
            self.compute_client.register_container(full_image_uri, "docker")
        )

        metadata = extract_metadata_from_image(docker_client, image)
        common_steps = metadata.pop("steps")

        for key, record in metadata.items():
            if "." in key:
                # skip "{key}.garden_doi" and "{key}.entrypoint_step" for now
                continue

            # register function & populate RegisteredEntrypoint fields
            to_register = make_function_to_register(key)
            record["container_uuid"] = container_uuid
            record["func_uuid"] = self.compute_client.register_function(
                to_register, container_uuid=str(container_uuid), public=True
            )
            entrypoint_step = metadata.get(f"{key}.entrypoint_step")
            all_steps = common_steps[:]
            all_steps.append(entrypoint_step)
            record["steps"] = all_steps
            record["doi"] = record.get("doi") or self._mint_draft_doi()
            record["short_name"] = record.get("short_name") or key
            record["notebook_url"] = notebook_url
            record["base_image_uri"] = base_image_uri
            record["full_image_uri"] = full_image_uri

            registered = RegisteredEntrypoint(**record)
            self._update_datacite(registered)
            local_data.put_local_entrypoint(registered)

            # fetch garden we're attaching this entrypoint to (if one was specified)
            garden_doi = metadata.get(f"{key}.garden_doi")
            if garden_doi:
                garden = local_data.get_local_garden_by_doi(garden_doi)
                if garden is None:
                    msg = (
                        f"Could not add entrypoint {key} to garden "
                        f"{garden_doi}: could not find local garden with that DOI"
                    )
                    raise ValueError(msg)
                garden.add_entrypoint(registered.doi, replace=True)
                local_data.put_local_garden(garden)
                dirty_gardens |= {garden.doi}
                print(
                    f"Added entrypoint {registered.doi} ({registered.short_name}) to garden {garden.doi} ({garden.title})!"
                )

        for doi in dirty_gardens:
            garden = local_data.get_local_garden_by_doi(doi)
            if garden:
                print(f"(Re-)publishing garden {garden.doi} ({garden.title}) ...")
                self.publish_garden_metadata(garden)
