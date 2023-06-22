import shutil

import requests

from garden_ai.mlmodel import LocalModel
from garden_ai.backend_client import (
    BackendClient,
    PresignedUrlDirection,
    PresignedUrlResponse,
)


def upload_mlmodel_to_s3(
    local_directory: str, local_model: LocalModel, backend_client: BackendClient
):
    # Get url from Garden API
    model_storage_path = f"{local_model.user_email}/{local_model.model_name}/model.zip"
    presigned_url_response = backend_client.get_presigned_url(
        model_storage_path, PresignedUrlDirection.Upload
    )
    _upload_directory_to_s3_presigned(local_directory, presigned_url_response)


def _upload_directory_to_s3_presigned(
    local_directory: str, presigned_url: PresignedUrlResponse
):
    """
    Zip a local directory and upload it to an S3 bucket using a presigned URL.

    Parameters
    ----------
    local_directory : str
        The local directory to upload.
    presigned_url: PresignedUrlResponse
        The url and fields needed to upload, generated by the backend
    """
    zip_filepath = shutil.make_archive(
        "model.zip", "zip", local_directory
    )  # local_directory is root dir, so it should be relative to that

    with open(zip_filepath, "rb") as f:
        files = {"file": ("model.zip", f)}
        http_response = requests.post(
            presigned_url.url, data=presigned_url.fields, files=files
        )

    # If successful, returns HTTP status code 204
    if http_response.status_code != 204:
        raise Exception(
            f"Failed to upload model directory {local_directory} to Garden. {http_response.text}"
        )