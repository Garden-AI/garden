import logging

import globus_sdk
from globus_sdk.scopes import AuthScopes, SearchScopes
from funcx.sdk.web_client import FuncxWebClient  # type: ignore
from funcx import FuncXClient  # type: ignore

from typing import Union, Dict

logger = logging.getLogger()


class FuncXLoginManager:
    """
    Implements the funcx.sdk.login_manager.protocol.LoginManagerProtocol class.
    https://github.com/funcx-faas/funcX/blob/main/funcx_sdk/funcx/sdk/login_manager/protocol.py#L18
    """

    def __init__(self, authorizers: Dict[str, globus_sdk.RefreshTokenAuthorizer]):
        self.authorizers = authorizers

    def get_auth_client(self) -> globus_sdk.AuthClient:
        return globus_sdk.AuthClient(authorizer=self.authorizers[AuthScopes.openid])

    def get_search_client(self) -> globus_sdk.SearchClient:
        return globus_sdk.SearchClient(authorizer=self.authorizers[SearchScopes.all])

    def get_funcx_web_client(
        self,
        *,
        base_url: Union[str, None] = None,
        app_name: Union[str, None] = None,
    ) -> FuncxWebClient:
        return FuncxWebClient(
            base_url=base_url,
            app_name=app_name,
            authorizer=self.authorizers[FuncXClient.FUNCX_SCOPE],
        )

    def ensure_logged_in(self):
        logger.warning("ensure_logged_in cannot be invoked from here!")

    def logout(self):
        logger.warning("logout cannot be invoked from here!")
