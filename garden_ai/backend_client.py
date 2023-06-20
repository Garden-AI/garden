import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional

import requests

from garden_ai.constants import GardenConstants

logger = logging.getLogger()


@dataclass
class PresignedUrlResponse:
    url: str
    fields: Optional[dict]  # Present for upload URLs. Absent for download URLs.


class PresignedUrlDirection(Enum):
    Upload = "upload"
    Download = "download"


# Client for the Garden backend API. The name "GardenClient" was taken :)
class BackendClient:
    def __init__(self, garden_authorizer):
        self.garden_authorizer = garden_authorizer

    def _call(self, http_verb, resource, payload) -> dict:
        headers = {
            "Content-Type": "application/vnd.api+json",
            "Authorization": self.garden_authorizer.get_authorization_header(),
        }
        url = GardenConstants.GARDEN_ENDPOINT + resource
        resp = http_verb(url, headers=headers, json=payload)
        try:
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError:
            logger.error(f"{resp.text}")
            raise
        except requests.exceptions.JSONDecodeError:
            logger.error(f"Could not parse response as JSON. {resp.text}")
            raise

    def _post(self, resource, payload) -> dict:
        return self._call(requests.post, resource, payload)

    def _put(self, resource, payload) -> dict:
        return self._call(requests.put, resource, payload)

    def mint_doi_on_datacite(self, payload: dict) -> str:
        response_dict = self._post("/doi", payload)
        doi = response_dict.get("doi", None)
        if not doi:
            raise Exception("Failed to mint DOI. Response was missing doi field.")
        return doi

    def do_search_thing(self):
        pass

    def get_presigned_url(self, full_model_name: str, direction: PresignedUrlDirection):
        payload = {"s3_path": full_model_name, "direction": direction.value}
        response_dict = self._post("/presigned-url", payload)
        url = response_dict.get("url", None)
        fields = response_dict.get("fields", None)
        if not url:
            raise Exception(
                "Failed to generate presigned URL for model file transfer. Response was missing url field."
            )
        if direction == PresignedUrlDirection.Upload and not fields:
            message = "Failed to generate presigned URL for model file upload. Response was missing 'fields' attribute."
            raise Exception(message)
        return PresignedUrlResponse(url, fields)
