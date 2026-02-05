import logging
import time
from typing import Callable

import requests

from garden_ai.constants import GardenConstants
from garden_ai.gardens import Garden
from garden_ai.schemas.garden import (
    GardenCreateRequest,
    GardenMetadata,
    GardenPatchRequest,
)
from garden_ai.schemas.groundhog import (
    HpcEndpointCreateRequest,
    HpcEndpointPatchRequest,
    HpcEndpointResponse,
    HpcFunctionCreateRequest,
    HpcFunctionPatchRequest,
    HpcFunctionResponse,
)
from garden_ai.schemas.hpc import HpcInvocationCreateRequest
from garden_ai.schemas.modal import (
    ModalBlobUploadURLRequest,
    ModalBlobUploadURLResponse,
    ModalInvocationRequest,
    ModalInvocationResponse,
)
from garden_ai.schemas.modal_app import (
    ModalAppCreateRequest,
    ModalAppPatchRequest,
    ModalAppResponse,
    ModalFunctionPatchRequest,
    ModalFunctionResponse,
)

logger = logging.getLogger()


# Client for the Garden backend API. The name "GardenClient" was taken :)
class BackendClient:
    def __init__(self, garden_authorizer):
        self.garden_authorizer = garden_authorizer

    def _call(
        self,
        http_verb: Callable,
        resource: str,
        payload: dict | None,
        params: dict | None = None,
    ) -> dict:
        headers = {"Authorization": self.garden_authorizer.get_authorization_header()}
        url = GardenConstants.GARDEN_ENDPOINT + resource
        if params:
            resp = http_verb(url, headers=headers, json=payload, params=params)
        else:
            resp = http_verb(url, headers=headers, json=payload)
        try:
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError:
            logger.error(
                f"Request to Garden backend failed. Status code {resp.status_code}. {resp.text}"
            )
            raise
        except requests.exceptions.JSONDecodeError:
            logger.error(f"Could not parse response as JSON. {resp.text}")
            raise

    def _post(self, resource: str, payload: dict) -> dict:
        return self._call(requests.post, resource, payload)

    def _put(self, resource: str, payload: dict) -> dict:
        return self._call(requests.put, resource, payload)

    def _delete(self, resource: str, payload: dict) -> dict:
        return self._call(requests.delete, resource, payload)

    def _get(self, resource: str, params: dict | None = None) -> dict:
        return self._call(requests.get, resource, None, params=params)

    def get_garden(self, doi: str) -> Garden:
        response = self._get(f"/gardens/{doi}")
        if response.get("is_archived", True):
            raise Exception(f"Garden with DOI {doi} is archived.")
        return Garden._from_nested_metadata(response)

    def get_garden_metadata(self, doi: str) -> GardenMetadata:
        # like get_garden but returns metadata only
        result = self._get(f"/gardens/{doi}")
        return GardenMetadata(**result)

    def delete_garden(self, doi: str):
        self._delete(f"/gardens/{doi}", {})

    def get_gardens(
        self,
        dois: list[str] | None = None,
        tags: list[str] | None = None,
        draft: bool | None = None,
        authors: list[str] | None = None,
        contributors: list[str] | None = None,
        year: str | None = None,
        owner_uuid: str | None = None,
        limit: int = 50,
    ) -> list[Garden]:
        params = {
            "doi": dois,
            "draft": draft,
            "owner_uuid": owner_uuid,
            "authors": authors,
            "contributors": contributors,
            "tags": tags,
            "year": year,
            "limit": limit,
        }
        params = {k: v for k, v in params.items() if v is not None}

        result = self._get("/gardens", params=params)

        gardens = []
        for data in result:
            gardens += [Garden._from_nested_metadata(data)]
        return gardens

    def get_user_info(self) -> dict:
        return self._get("/users")

    def invoke_modal_function(
        self,
        payload: ModalInvocationRequest,
    ) -> ModalInvocationResponse:
        response = self._post("/modal-invocations", payload.model_dump(mode="json"))
        return ModalInvocationResponse(**response)

    def invoke_modal_function_async(
        self,
        payload: ModalInvocationRequest,
    ) -> ModalInvocationResponse:
        invocation_response = self._post(
            "/modal-invocations/async", payload.model_dump(mode="json")
        )
        output_response = self._get(f"/modal-invocations/{invocation_response['id']}")

        start_time = time.time()
        retry_count = 0
        while output_response["status"] == "pending":
            elapsed = time.time() - start_time
            if elapsed > 30:
                poll_interval = 5.0
            else:
                poll_interval = GardenConstants.BACKEND_POLL_INTERVAL_SECONDS

            time.sleep(poll_interval)

            try:
                output_response = self._get(
                    f"/modal-invocations/{invocation_response['id']}"
                )
                retry_count = 0
            except requests.HTTPError as e:
                if (
                    e.response.status_code == 502 or e.response.status_code == 504
                ) and retry_count <= 3:
                    retry_count += 1
                    # If we are in the fast polling phase (elapsed < 30),
                    # we want to wait a bit longer (e.g. 1s) before retrying/polling again.
                    # If we are in the slow polling phase (elapsed > 30),
                    # the next loop iteration will wait 5s anyway.
                    if elapsed <= 30:
                        time.sleep(1.0)
                    continue
                else:
                    raise e

        match output_response["status"]:
            case "done":
                return ModalInvocationResponse(
                    data_format=1, result=output_response["result"]
                )
            case _:
                raise Exception(
                    f"Error invoking Modal function with id: {invocation_response['id']}:\n\t"
                    f"Invocation Status: {output_response['status']}\n\t"
                    f"Error: {output_response['error']}"
                )

    def get_blob_upload_url(
        self, payload: ModalBlobUploadURLRequest
    ) -> ModalBlobUploadURLResponse:
        response = self._post(
            "/modal-invocations/blob-uploads", payload.model_dump(mode="json")
        )
        return ModalBlobUploadURLResponse(**response)

    def search_gardens(self, payload: dict) -> dict:
        result = self._post("/gardens/search", payload=payload)

        return result

    def create_hpc_invocation(self, payload: HpcInvocationCreateRequest) -> dict:
        response = self._post("/hpc/invocations", payload.model_dump(mode="json"))
        return response

    # =========================================================================
    # Garden CRUD Methods
    # =========================================================================

    def create_garden(self, payload: GardenCreateRequest) -> GardenMetadata:
        """Create a new garden."""
        response = self._post("/gardens", payload.model_dump(mode="json"))
        return GardenMetadata(**response)

    def patch_garden(self, doi: str, payload: GardenPatchRequest) -> GardenMetadata:
        """Update an existing garden."""
        response = self._call(
            requests.patch,
            f"/gardens/{doi}",
            payload.model_dump(mode="json", exclude_none=True),
        )
        return GardenMetadata(**response)

    # =========================================================================
    # Modal App CRUD Methods
    # =========================================================================

    def create_modal_app(self, payload: ModalAppCreateRequest) -> ModalAppResponse:
        """Deploy a Modal app synchronously."""
        response = self._post("/modal-apps", payload.model_dump(mode="json"))
        return ModalAppResponse(**response)

    def create_modal_app_async(
        self, payload: ModalAppCreateRequest
    ) -> ModalAppResponse:
        """Deploy a Modal app asynchronously. Returns immediately with pending status."""
        response = self._post("/modal-apps/async", payload.model_dump(mode="json"))
        return ModalAppResponse(**response)

    def get_modal_app(self, app_id: int) -> ModalAppResponse:
        """Get a Modal app by ID."""
        response = self._get(f"/modal-apps/{app_id}")
        return ModalAppResponse(**response)

    def get_modal_apps(self) -> list[ModalAppResponse]:
        """List all Modal apps for the current user."""
        response = self._get("/modal-apps/")
        return [ModalAppResponse(**app) for app in response]

    def patch_modal_app(
        self, app_id: int, payload: ModalAppPatchRequest
    ) -> ModalAppResponse:
        """Update a Modal app. Triggers redeployment if file_contents changed."""
        response = self._call(
            requests.patch,
            f"/modal-apps/async/{app_id}",
            payload.model_dump(mode="json", exclude_none=True),
        )
        return ModalAppResponse(**response)

    def delete_modal_app(self, app_id: int) -> dict:
        """Delete a Modal app."""
        return self._delete(f"/modal-apps/{app_id}", {})

    def poll_modal_app_deployment(
        self, app_id: int, timeout: float = 300.0, poll_interval: float = 2.0
    ) -> ModalAppResponse:
        """Poll a Modal app until deployment completes or fails."""
        start_time = time.time()
        while True:
            app = self.get_modal_app(app_id)
            if app.deploy_status in ("success", "failed", None):
                return app
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Modal app deployment timed out after {timeout}s. "
                    f"Current status: {app.deploy_status}"
                )
            time.sleep(poll_interval)

    # =========================================================================
    # Modal Function Methods
    # =========================================================================

    def get_modal_function(self, function_id: int) -> ModalFunctionResponse:
        """Get a Modal function by ID."""
        response = self._get(f"/modal-functions/{function_id}")
        return ModalFunctionResponse(**response)

    def get_modal_functions(
        self,
        ids: list[int] | None = None,
        tags: list[str] | None = None,
        authors: list[str] | None = None,
        owner_uuid: str | None = None,
        draft: bool | None = None,
        year: str | None = None,
        limit: int = 50,
    ) -> list[ModalFunctionResponse]:
        """List Modal functions with optional filters."""
        params = {
            "id": ids,
            "tags": tags,
            "authors": authors,
            "owner_uuid": owner_uuid,
            "draft": draft,
            "year": year,
            "limit": limit,
        }
        params = {k: v for k, v in params.items() if v is not None}
        response = self._get("/modal-functions", params=params)
        return [ModalFunctionResponse(**fn) for fn in response]

    def patch_modal_function(
        self, function_id: int, payload: ModalFunctionPatchRequest
    ) -> ModalFunctionResponse:
        """Update a Modal function's metadata."""
        response = self._call(
            requests.patch,
            f"/modal-functions/{function_id}",
            payload.model_dump(mode="json", exclude_none=True),
        )
        return ModalFunctionResponse(**response)

    # =========================================================================
    # HPC/Groundhog Endpoint Methods
    # =========================================================================

    def create_hpc_endpoint(
        self, payload: HpcEndpointCreateRequest
    ) -> HpcEndpointResponse:
        """Create a new HPC endpoint."""
        response = self._post("/hpc/endpoints", payload.model_dump(mode="json"))
        return HpcEndpointResponse(**response)

    def get_hpc_endpoint(self, endpoint_id: int) -> HpcEndpointResponse:
        """Get an HPC endpoint by ID."""
        response = self._get(f"/hpc/endpoints/{endpoint_id}")
        return HpcEndpointResponse(**response)

    def get_hpc_endpoints(self, limit: int = 50) -> list[HpcEndpointResponse]:
        """List all HPC endpoints."""
        response = self._get("/hpc/endpoints", params={"limit": limit})
        return [HpcEndpointResponse(**ep) for ep in response]

    def patch_hpc_endpoint(
        self, endpoint_id: int, payload: HpcEndpointPatchRequest
    ) -> HpcEndpointResponse:
        """Update an HPC endpoint."""
        response = self._call(
            requests.patch,
            f"/hpc/endpoints/{endpoint_id}",
            payload.model_dump(mode="json", exclude_none=True),
        )
        return HpcEndpointResponse(**response)

    def delete_hpc_endpoint(self, endpoint_id: int) -> dict:
        """Delete an HPC endpoint."""
        return self._delete(f"/hpc/endpoints/{endpoint_id}", {})

    # =========================================================================
    # HPC/Groundhog Function Methods
    # =========================================================================

    def create_hpc_function(
        self, payload: HpcFunctionCreateRequest
    ) -> HpcFunctionResponse:
        """Create a new HPC function."""
        response = self._post("/hpc/functions", payload.model_dump(mode="json"))
        return HpcFunctionResponse(**response)

    def get_hpc_function(self, function_id: int) -> HpcFunctionResponse:
        """Get an HPC function by ID."""
        response = self._get(f"/hpc/functions/{function_id}")
        return HpcFunctionResponse(**response)

    def get_hpc_functions(self) -> list[HpcFunctionResponse]:
        """List HPC functions for the current user."""
        response = self._get("/hpc/functions")
        return [HpcFunctionResponse(**fn) for fn in response]

    def patch_hpc_function(
        self, function_id: int, payload: HpcFunctionPatchRequest
    ) -> HpcFunctionResponse:
        """Update an HPC function."""
        response = self._call(
            requests.patch,
            f"/hpc/functions/{function_id}",
            payload.model_dump(mode="json", exclude_none=True),
        )
        return HpcFunctionResponse(**response)

    def delete_hpc_function(self, function_id: int) -> dict:
        """Delete an HPC function."""
        return self._delete(f"/hpc/functions/{function_id}", {})
