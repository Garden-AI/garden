import logging
import os
import sys
from pathlib import Path

from globus_sdk import GroupsClient, RefreshTokenAuthorizer, NativeAppAuthClient, \
    AuthClient, AuthAPIError
from globus_sdk.tokenstorage import SimpleJSONFileAdapter

logger = logging.getLogger()


class GardenClient:
    """
    Main class for interacting with the Garden service
    Holds helper operations for performing common tasks
    with the Garden service.

    Will authenticate with GlobusAuth, storing generated keys in the users .garden
    directory
    """

    def __init__(self, auth_client: AuthClient = None):
        key_store_path = Path(os.path.expanduser("~/.garden"))
        key_store_path.mkdir(exist_ok=True)
        self.auth_key_store = SimpleJSONFileAdapter(os.path.join(key_store_path, "tokens.json"))
        self.client_id = os.environ.get("GARDEN_CLIENT_ID", "cf9f8938-fb72-439c-a70b-85addf1b8539")

        self.auth_client = NativeAppAuthClient(self.client_id) \
            if not auth_client\
            else auth_client

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
                sys.exit(-1)

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
