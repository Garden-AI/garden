import logging
import time
from typing import Callable

import requests

from garden_ai.constants import GardenConstants
from garden_ai.schemas.garden import GardenMetadata
from garden_ai.schemas.modal import (
    ModalInvocationResponse,
    ModalInvocationRequest,
    ModalBlobUploadURLRequest,
    ModalBlobUploadURLResponse,
)
from garden_ai.gardens import Garden

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

        while output_response["status"] == "pending":
            time.sleep(GardenConstants.BACKEND_POLL_INTERVAL_SECONDS)
            output_response = self._get(
                f"/modal-invocations/{invocation_response['id']}"
            )

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
