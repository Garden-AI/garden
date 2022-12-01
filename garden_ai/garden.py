import logging
import os
from pathlib import Path

from globus_sdk import (
    GroupsClient,
    RefreshTokenAuthorizer,
    NativeAppAuthClient,
    AuthClient,
    AuthAPIError,
)
from typing import List
from globus_sdk.tokenstorage import SimpleJSONFileAdapter

from garden_ai.model import Garden
from pydantic import ValidationError

logger = logging.getLogger()


class AuthException(Exception):
    pass


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

    def __init__(self, auth_client: AuthClient = None):
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

    def _do_login_flow(self):
        self.auth_client.oauth2_start_flow(
            requested_scopes=GroupsClient.scopes.view_my_groups_and_memberships,
            refresh_tokens=True,
        )
        authorize_url = self.auth_client.oauth2_get_authorize_url()
        print(f"Please go to this URL and login:\n\n{authorize_url}\n")
        auth_code = input("Please enter the code here: ").strip()

        try:
            tokens = self.auth_client.oauth2_exchange_code_for_tokens(auth_code)
            return tokens
        except AuthAPIError:
            logger.fatal("Invalid Globus auth token received. Exiting")
            return None

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

    def create_garden(self, authors: List[str] = [], title: str = "", **kwargs):
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

    def register_metadata(self, garden: Garden, out_dir=None):
        """Make a Garden object's metadata discoverable.

        This will perform validation on the metadata fields and (if successful)
        write the metadata to a `"metadata.json"` file in the current working directory.

        Parameters
        ----------
        garden : Garden
            A Garden object with user's ready-to-publish metadata. Users might
            want to invoke `to_do()` method on their Garden, or see `Garden`
            docstring for explanation of any unset required/recommended fields.

        out_dir : Union[PathLike, str]
            Directory in which a copy of the Garden's metadata is written on
            successful registration. Defaults to current working directory.

        Raises
        ------
        ValidationError

        """
        out_dir = Path(out_dir) if out_dir else Path.cwd()
        out_dir /= "metadata.json"
        try:
            garden.request_doi()
            garden.validate()
        except ValidationError as e:
            logger.error(e)
            raise
        else:
            with open(out_dir, "w+") as f:
                f.write(garden.json())
