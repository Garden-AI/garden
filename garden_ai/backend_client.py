import json
import logging
from typing import Callable, Optional
import boto3

import requests

from garden_ai.constants import GardenConstants
from garden_ai.gardens import Garden, PublishedGarden
from garden_ai.entrypoints import RegisteredEntrypoint

logger = logging.getLogger()


# Client for the Garden backend API. The name "GardenClient" was taken :)
class BackendClient:
    def __init__(self, garden_authorizer):
        self.garden_authorizer = garden_authorizer

    def _call(
        self, http_verb: Callable, resource: str, payload: Optional[dict]
    ) -> dict:
        headers = {"Authorization": self.garden_authorizer.get_authorization_header()}
        url = GardenConstants.GARDEN_ENDPOINT + resource
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

    def _get(self, resource: str) -> dict:
        return self._call(requests.get, resource, None)

    def mint_doi_on_datacite(self, payload: dict) -> str:
        response_dict = self._post("/doi", payload)
        doi = response_dict.get("doi", None)
        if not doi:
            raise ValueError("Failed to mint DOI. Response was missing doi field.")
        return doi

    def update_doi_on_datacite(self, payload: dict):
        self._put("/doi", payload)

    def publish_garden_metadata(self, garden: PublishedGarden):
        payload = json.loads(garden.json())
        self._post("/garden-search-record", payload)

    def delete_garden_metadata(self, doi: str):
        self._delete("/garden-search-record", {"doi": doi})

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

    def create_garden(self, garden: Garden):
        self._post("/gardens", garden.model_dump(mode="json"))
        return

    def update_garden(self, garden: Garden) -> PublishedGarden:
        doi = garden.doi
        result = self._put(f"/gardens/{doi}", garden.model_dump(mode="json"))
        return PublishedGarden(**result)

    def get_garden(self, doi: str) -> PublishedGarden:
        result = self._get(f"/gardens/{doi}")
        return PublishedGarden(**result)

    def delete_garden(self, doi: str):
        self._delete(f"/gardens/{doi}", {})

    def create_entrypoint(self, entrypoint: RegisteredEntrypoint):
        if not entrypoint.function_text:
            entrypoint.function_text = entrypoint.steps[0].function_text
        payload = entrypoint.model_dump(mode="json", exclude={"steps"})
        self._post("/entrypoints", payload)

    def update_entrypoint(self, entrypoint: RegisteredEntrypoint):
        doi = entrypoint.doi
        if not entrypoint.function_text:
            entrypoint.function_text = entrypoint.steps[0].function_text
        payload = entrypoint.model_dump(mode="json", exclude={"steps"})
        self._put(f"/entrypoints/{doi}", payload)

    def get_entrypoint(self, doi: str) -> RegisteredEntrypoint:
        result = self._get(f"/entrypoints/{doi}")
        return RegisteredEntrypoint(**result)

    def delete_entrypoint(self, doi: str):
        self._delete(f"/entrypoints/{doi}", {})
