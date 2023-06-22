from garden_ai.mlmodel import MODEL_STAGING_DIR
from garden_ai.backend_client import PresignedUrlDirection, BackendClient


def generate_download_url(backend_client: BackendClient, model_full_name: str):
    pass


def download_and_stage(presigned_download_url: str):
    # Download and unzip into MODEL_STAGING_DIR
    pass
