import logging
from typing import Dict, Union

import globus_sdk
from globus_compute_sdk import Client  # type: ignore
from globus_compute_sdk.sdk.web_client import WebClient  # type: ignore
from globus_sdk.scopes import AuthScopes, SearchScopes

logger = logging.getLogger()


class ComputeLoginManager:
    """
    Implements the globus_compute_sdk.sdk.login_manager.protocol.LoginManagerProtocol class.
    https://github.com/funcx-faas/funcX/blob/main/funcx_sdk/funcx/sdk/login_manager/protocol.py#L18
    """

    def __init__(
        self,
        authorizers: Dict[
            str,
            Union[
                globus_sdk.RefreshTokenAuthorizer,
                globus_sdk.ClientCredentialsAuthorizer,
            ],
        ],
    ):
        self.authorizers = authorizers

    def get_auth_client(self) -> globus_sdk.AuthClient:
        return globus_sdk.AuthClient(authorizer=self.authorizers[AuthScopes.openid])

    def get_search_client(self) -> globus_sdk.SearchClient:
        return globus_sdk.SearchClient(authorizer=self.authorizers[SearchScopes.all])

    def get_web_client(
        self,
        *,
        base_url: Union[str, None] = None,
        app_name: Union[str, None] = None,
    ) -> WebClient:
        return WebClient(
            base_url=base_url,
            app_name=app_name,
            authorizer=self.authorizers[Client.FUNCX_SCOPE],
        )

    def ensure_logged_in(self):
        raise NotImplementedError("ensure_logged_in cannot be invoked from here")

    def logout(self):
        raise NotImplementedError("logout cannot be invoked from here")
