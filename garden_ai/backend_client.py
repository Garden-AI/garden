import json
import logging
from typing import Callable

import boto3
import requests

from garden_ai.constants import GardenConstants
from garden_ai.schemas.entrypoint import (
    RegisteredEntrypointMetadata,
)
from garden_ai.schemas.garden import GardenMetadata
from garden_ai.schemas.modal import ModalInvocationResponse, ModalInvocationRequest
from garden_ai.entrypoints import Entrypoint
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

    def mint_doi_on_datacite(self, payload: dict) -> str:
        response_dict = self._post("/doi", payload)
        doi = response_dict.get("doi", None)
        if not doi:
            raise ValueError("Failed to mint DOI. Response was missing doi field.")
        return doi

    def update_doi_on_datacite(self, payload: dict):
        self._put("/doi", payload)

    def upload_notebook(
        self, notebook_contents: dict, username: str, notebook_name: str
    ):
        payload = {
            "notebook_json": json.dumps(notebook_contents),
            "notebook_name": notebook_name,
            "folder": username,
        }
        resp = self._post("/notebook", payload)
        return resp["notebook_url"]

    def get_docker_push_session(self) -> boto3.Session:
        resp = self._get("/docker-push-token")

        # Make sure the response has the expected fields
        for field in ["AccessKeyId", "SecretAccessKey", "SessionToken", "ECRRepo"]:
            if field not in resp or not resp[field]:
                raise ValueError(
                    f"/docker-push-token response missing field {field}. Full response: {resp}"
                )

        return boto3.Session(
            aws_access_key_id=resp["AccessKeyId"],
            aws_secret_access_key=resp["SecretAccessKey"],
            aws_session_token=resp["SessionToken"],
            region_name="us-east-1",
        )

    def get_garden(self, doi: str) -> Garden:
        response = self._get(f"/gardens/{doi}")
        return Garden._from_nested_metadata(response)

    def put_garden(self, garden_meta: GardenMetadata) -> Garden:
        doi = garden_meta.doi
        response = self._put(f"/gardens/{doi}", garden_meta.model_dump(mode="json"))
        return Garden._from_nested_metadata(response)

    def get_garden_metadata(self, doi: str) -> GardenMetadata:
        # like get_garden but returns metadata only
        result = self._get(f"/gardens/{doi}")
        return GardenMetadata(**result)

    def delete_garden(self, doi: str):
        self._delete(f"/gardens/{doi}", {})

    def get_entrypoint_metadata(self, doi: str) -> RegisteredEntrypointMetadata:
        result = self._get(f"/entrypoints/{doi}")
        return RegisteredEntrypointMetadata(**result)

    def put_entrypoint_metadata(
        self, entrypoint_meta: RegisteredEntrypointMetadata
    ) -> RegisteredEntrypointMetadata:
        doi = entrypoint_meta.doi
        response = self._put(
            f"/entrypoints/{doi}", entrypoint_meta.model_dump(mode="json")
        )
        updated_entrypoint = RegisteredEntrypointMetadata(**response)
        return updated_entrypoint

    def get_entrypoint(self, doi: str) -> Entrypoint:
        # like get_entrypoint_metadata, but returns the callable object
        result = self._get(f"/entrypoints/{doi}")
        return Entrypoint(RegisteredEntrypointMetadata(**result))

    def delete_entrypoint(self, doi: str):
        self._delete(f"/entrypoints/{doi}", {})

    def get_entrypoints(
        self,
        dois: list[str] | None = None,
        tags: list[str] | None = None,
        authors: list[str] | None = None,
        draft: bool | None = None,
        year: str | None = None,
        owner_uuid: str | None = None,
        limit: int = 50,
    ) -> list[Entrypoint]:
        params = {
            "doi": dois,
            "tags": tags,
            "authors": authors,
            "owner_uuid": owner_uuid,
            "draft": draft,
            "year": year,
            "limit": limit,
        }
        # skip unspecified values
        params = {k: v for k, v in params.items() if v is not None}

        response: list[dict] = self._get("/entrypoints", params=params)  # type: ignore

        entrypoints = [
            Entrypoint(RegisteredEntrypointMetadata(**data)) for data in response
        ]
        return entrypoints

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
