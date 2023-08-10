# mypy: disable-error-code="import"
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Union
from uuid import UUID

import typer
from globus_compute_sdk import Client
from globus_compute_sdk.serialize.concretes import DillCode
from globus_compute_sdk.sdk.login_manager.tokenstore import get_token_storage_adapter
from garden_ai.garden_file_adapter import GardenFileAdapter
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
from garden_ai.gardens import Garden, PublishedGarden
from garden_ai.globus_compute.containers import build_container
from garden_ai.globus_compute.remote_functions import register_pipeline
from garden_ai.globus_search import garden_search
from garden_ai.local_data import (
    GardenNotFoundException,
    PipelineNotFoundException,
    _read_local_cache,
    _write_local_cache,
)
from garden_ai.mlmodel import (
    LocalModel,
    ModelMetadata,
    DatasetConnection,
    ModelNotFoundException,
    Model,
    clear_mlflow_staging_directory,
    stage_model_for_upload,
)
from garden_ai.model_file_transfer.upload import upload_mlmodel_to_s3
from garden_ai.pipelines import Pipeline, RegisteredPipeline, Paper, Repository
from garden_ai.steps import step
from garden_ai.utils.misc import extract_email_from_globus_jwt, get_cache_tag


GARDEN_ENDPOINT = os.environ.get(
    "GARDEN_ENDPOINT",
    "https://nu3cetwc84.execute-api.us-east-1.amazonaws.com/garden_prod",
)

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
        self.garden_key_store = SimpleJSONFileAdapter(
            os.path.join(key_store_path, "tokens.json")
        )
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
            code_serialization_strategy=DillCode(),
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

    def create_pipeline(
        self, authors: Optional[List[str]] = None, title: Optional[str] = None, **kwargs
    ) -> Pipeline:
        """Initialize and return a pipeline object.

        If this pipeline's DOI has been used before to register a function for
        remote execution, reuse the (funcx/globus compute) ID for consistency.

        NOTE: this means that local modifications to a pipeline will not be
        reflected when executing remotely until the pipeline is re-registered.
        """
        data = dict(kwargs)
        if authors:
            data["authors"] = authors
        if title:
            data["title"] = title
        data["doi"] = data.get("doi") or self._mint_draft_doi()

        return Pipeline(**data)

    def register_model(self, local_model: LocalModel) -> ModelMetadata:
        try:
            # Create directory in MLModel format
            model_directory = stage_model_for_upload(local_model)
            # Push contents of directory to S3
            upload_mlmodel_to_s3(model_directory, local_model, self.backend_client)
        finally:
            try:
                clear_mlflow_staging_directory()
            except Exception as e:
                logger.error(
                    "Could not clean up model staging directory. Check permissions on ~/.garden/mlflow"
                )
                logger.error("Original exception: " + str(e))
                # We can still proceed, there is just some cruft in the user's home directory.
                pass
        registered_model = ModelMetadata(**local_model.dict())
        local_data.put_local_model(registered_model)
        return registered_model

    def add_dataset(self, model_name: str, title: str, url: str, **kwargs) -> None:
        """Adds a ``DatasetConnection`` to ``ModelMetadata`` corresponding to the given full model name.

        Parameters
        ----------
        model_name : str
            The previously registered model's full model name. Raises an
            exception if not found.

        title : str
            An official name or title for the dataset.

        url: str
            The url to access this dataset.

        **kwargs :
            Metadata for the new DatasetConnection object. Keyword arguments matching
            required or recommended fields will be (where necessary) coerced to the appropriate type.
            May include: Optional[str] doi, Optional[str] data_type.

        Raises
        ------
        ModelNotFoundException
            Raised when no known model exists with the given identifier.
        """
        model = local_data.get_local_model_by_name(model_name)
        data = dict(kwargs)
        if not model:
            raise ModelNotFoundException("This model could not be found")
        if model_name:
            data["model_name"] = model_name
        if title:
            data["title"] = title
        if url:
            data["url"] = url
        dataset = DatasetConnection(**data)
        model.dataset = dataset
        local_data.put_local_model(model)

    def _mint_draft_doi(self, test: bool = True) -> str:
        """Register a new draft DOI with DataCite via Garden backend.

        Expects environment variable GARDEN_ENDPOINT to be set (not including `/doi`).

        Parameters
        ----------
        test : bool
            toggle which garden backend endpoint to hit; we do not yet have a
            test endpoint so test=False raises NotImplementedError.

        Raises
        ------
        NotImplementedError
            see `test`
        """

        if not test:
            raise NotImplementedError

        logger.info("Requesting draft DOI")
        payload = {
            "data": {"type": "dois", "attributes": {}}
        }  # required data is filled in on the backend
        return self.backend_client.mint_doi_on_datacite(payload)

    def _update_datacite(self, obj: Union[PublishedGarden, RegisteredPipeline]) -> None:
        logger.info("Requesting update to DOI")
        metadata = json.loads(obj.datacite_json())
        metadata.update(event="publish", url=f"https://thegardens.ai/{obj.doi}")

        payload = {"data": {"type": "dois", "attributes": metadata}}
        self.backend_client.update_doi_on_datacite(payload)
        logger.info("Update request succeeded")

    def build_container(self, pipeline: Pipeline) -> str:
        built_container_uuid = build_container(self.compute_client, pipeline)

        cache = _read_local_cache()
        tag = get_cache_tag(
            pipeline.pip_dependencies,
            pipeline.conda_dependencies,
            pipeline.python_version,
        )
        cache[tag] = built_container_uuid
        _write_local_cache(cache)

        return built_container_uuid

    def register_pipeline(
        self, pipeline: Pipeline, container_uuid: Optional[str] = None
    ) -> RegisteredPipeline:
        if container_uuid is None:
            cache = _read_local_cache().get(
                get_cache_tag(
                    pipeline.pip_dependencies,
                    pipeline.conda_dependencies,
                    pipeline.python_version,
                )
            )
            if cache is not None:
                container_uuid = cache
                print("Cache hit! Using pre-built container.")
            else:
                container_uuid = self.build_container(pipeline)

        func_uuid = register_pipeline(self.compute_client, pipeline, container_uuid)
        pipeline.func_uuid = UUID(func_uuid)
        registered = RegisteredPipeline.from_pipeline(pipeline)
        self._update_datacite(registered)
        local_data.put_local_pipeline(registered)
        return registered

    def add_paper(self, title: str, doi: str, **kwargs) -> None:
        """Adds a ``Paper`` to a ``RegisteredPipeline`` corresponding to the given DOI.

        Parameters
        ----------
        doi : str
            The previously registered pipeline's DOI. Raises an
            exception if not found.

        title : str
            An official name or title for the paper.

        **kwargs :
            Metadata for the new Paper object. Keyword arguments matching
            required or recommended fields will be (where necessary) coerced to the appropriate type.
            May include: List[str] authors and Optional[str] citation.

        Raises
        ------
        PipelineNotFoundException
            Raised when no known pipeline exists with the given identifier.
        """
        pipeline = local_data.get_local_pipeline_by_doi(doi)
        if not pipeline:
            raise PipelineNotFoundException(
                f"Could not find any pipelines with DOI {doi}."
            )
        data = dict(kwargs)
        if title:
            data["title"] = title
        paper = Paper(**data)
        pipeline.papers.append(paper)
        local_data.put_local_pipeline(pipeline)

    def add_repository(self, doi: str, url: str, repo_name: str, **kwargs) -> None:
        """Adds a ``Repository`` to a ``RegisteredPipeline`` corresponding to the given DOI.

        Parameters
        ----------
        doi : str
            The previously registered pipeline's DOI. Raises an
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
        PipelineNotFoundException
            Raised when no known pipeline exists with the given identifier.
        """
        data = dict(kwargs)
        if doi:
            data["doi"] = doi
        if url:
            data["url"] = url
        if repo_name:
            data["repo_name"] = repo_name
        pipeline = local_data.get_local_pipeline_by_doi(doi)
        if not pipeline:
            raise PipelineNotFoundException(
                f"Could not find any pipelines with DOI {doi}."
            )

        repository = Repository(**data)
        pipeline.repositories.append(repository)
        local_data.put_local_pipeline(pipeline)

    def add_simple_model_to_garden(
        self,
        local_model: LocalModel,
        garden_doi: str,
        *,
        input_type: type,
        output_type: type,
        title: str = None,
        description: str = None,
        **kwargs,
    ) -> str:
        """Take a `LocalModel` object and automatically perform all the necessary steps,
        using the additional provided information, to publish the model such that it is prepared to be run remotely.

        Parameters
        ----------
        local_model : LocalModel
            The model for which remote executiotability is desired. The object's necessary fields are documented at
            `~mlmodel.LocalModel` and its parent `~mlmodel.ModelMetadata`.

        garden_doi : str
            The DOI of a previously published Garden that the model will be added to in order to facilitate findability
            and remote execution.

        input_type : type
            The type that describes the expected input to your model.

        output_type : type
            The type that describes the expected output of your model.

        title : str
            Title that describes the pipeline that is implicitly generated. If no title is provided, one will be generated.

        description : str
            Description of the pipeline that is implicitly generated. If no description is provided, one will be generated.

        **kwargs :
            Metadata for the new Pipeline object that runs the model. Keyword arguments matching
            required or recommended fields will be (where necessary) coerced to the
            appropriate type and validated per the documentation found at `~pipelines.Pipeline`.
            NOTE: Some fields are necessary for proper publication. e.g. `pip_dependecies` if your model needs them.

        Returns
        -------
        The DOI of your auto-generated pipeline. Use this DOI to access your pipeline at a later date.

        Examples
        --------
        client = GardenClient()
        local_model = LocalModel(
            model_name="dendrite_segmentation",
            flavor="tensorflow",
            local_path="model.h5",
            user_email=client.get_email()
        )
        my_pipeline = client.add_simple_model_to_garden(local_model, "10.1234/doi-here",
                                                        input_type=np.ndarray,
                                                        output_type=np.ndarray,
                                                        authors=["Monty Python", "Guido van Rossum"],
                                                        pip_dependencies=["tensorflow"],
                                                        tags=["materials science", "computer vision"]
        )
        """
        registered_model = self.register_model(local_model)

        # we ignore these type errors, because we want the step to be typed correctly but mypy does not acknowledge that it would be
        @step
        def run_inference(
            input_arg: input_type, model=Model(registered_model.full_name)  # type: ignore
        ) -> output_type:  # type: ignore
            return model.predict(input_arg)

        pipeline = self.create_pipeline(
            title=title or f"Inference on model: {local_model.model_name}",
            short_name=local_model.model_name,
            steps=(run_inference,),
            description=description
            or "Auto-generated pipeline that executes a single step which runs an inference.",
            **kwargs,
        )
        registered = self.register_pipeline(pipeline)

        garden = self.clone_published_garden(garden_doi, silent=True)

        # NOTE hack to allow the clone to update the remote record in-place
        garden.doi = garden_doi

        # add pipeline to garden
        garden.add_pipeline(registered.doi)

        # update the record with new pipeline added
        self.publish_garden_metadata(garden)

        return registered.doi

    def get_registered_pipeline(self, doi: str) -> RegisteredPipeline:
        """Return a callable ``RegisteredPipeline`` corresponding to the given DOI.

        Parameters
        ----------
        doi : str
            The previously registered pipeline's DOI. Raises an
            exception if not found.

        Returns
        -------
        RegisteredPipeline
            Instance of ``RegisteredPipeline``, which can be run on
            a specified remote Globus Compute endpoint.

        Raises
        ------
        PipelineNotFoundException
            Raised when no known pipeline exists with the given identifier.
        """
        pipeline = local_data.get_local_pipeline_by_doi(doi)

        if pipeline is None:
            raise PipelineNotFoundException(
                f"Could not find any pipelines with DOI {doi}."
            )
        pipeline_url_json = self.generate_presigned_urls_for_pipeline(pipeline)
        pipeline._env_vars = {GardenConstants.URL_ENV_VAR_NAME: pipeline_url_json}
        return pipeline

    def get_email(self) -> str:
        return local_data._get_user_email()

    def get_local_garden(self, doi: str) -> Garden:
        """Return a registered ``Garden`` corresponding to the given DOI.

        Any ``RegisteredPipelines`` registered to the Garden will be callable
        as attributes on the garden by their (registered) short_name, e.g.
            ```python
                my_garden = client.get_local_garden('garden-doi')
                #  pipeline would have been registered with short_name='my_pipeline'
                my_garden.my_pipeline(*args, endpoint='where-to-execute')
            ```
        Tip: To access the pipeline by a different name, use ``my_garden.rename_pipeline(pipeline_id, new_name)``.
        To persist a new name for a pipeline, re-register it to the garden and specify an alias.

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

        for pipeline in published.pipelines:
            local_data.put_local_pipeline(pipeline)

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
        self._generate_presigned_urls_for_garden(garden)
        return garden

    def _generate_presigned_urls_for_garden(self, garden: PublishedGarden):
        all_model_names = [
            model_name
            for pipeline in garden.pipelines
            for model_name in pipeline.model_full_names
        ]  # flatten all model names
        all_presigned_urls = self.backend_client.get_model_download_urls(
            all_model_names
        )
        for pipeline in garden.pipelines:
            model_name_to_url = {
                presigned_url.model_name: presigned_url.url
                for presigned_url in all_presigned_urls
                if presigned_url.model_name in pipeline.model_full_names
            }
            pipeline._env_vars = {
                GardenConstants.URL_ENV_VAR_NAME: json.dumps(model_name_to_url)
            }

    def generate_presigned_urls_for_pipeline(
        self, pipeline: Union[RegisteredPipeline, Pipeline]
    ) -> str:
        all_presigned_urls = self.backend_client.get_model_download_urls(
            pipeline.model_full_names
        )
        model_name_to_url = {
            presigned_url.model_name: presigned_url.url
            for presigned_url in all_presigned_urls
        }
        return json.dumps(model_name_to_url)
