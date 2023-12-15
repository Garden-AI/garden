from globus_sdk.tokenstorage import SimpleJSONFileAdapter, SQLiteAdapter, FileAdapter
from globus_sdk.services.auth import OAuthTokenResponse

from typing import Any, Dict, Optional
from pathlib import Path


class GardenFileAdapter(FileAdapter):
    def __init__(
        self, garden_key_store: SimpleJSONFileAdapter, compute_key_store: SQLiteAdapter
    ):
        self.garden_key_store = garden_key_store
        self.compute_key_store = compute_key_store

    def store(self, token_response: OAuthTokenResponse) -> None:
        self.garden_key_store.store(token_response)
        self.compute_key_store.store(token_response)

    def get_token_data(self, resource_server: str) -> Optional[Dict[str, Any]]:
        return self.garden_key_store.get_token_data(resource_server)

    def get_by_resource_server(self) -> Dict[str, Any]:
        return self.garden_key_store.get_by_resource_server()

    def on_refresh(self, token_response: OAuthTokenResponse) -> None:
        self.store(token_response)

    def clear_garden_data(self) -> None:
        # silently ignores the case where the file is already gone
        Path.unlink(Path(self.garden_key_store.filename), missing_ok=True)

    def file_exists(self) -> bool:
        return self.garden_key_store.file_exists()
