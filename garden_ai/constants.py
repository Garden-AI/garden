import os
from dotenv import load_dotenv
import pathlib

# get env file from project root
dotenv_path = pathlib.Path(f"{__file__}/../..").resolve() / ".env.shared"
load_dotenv(str(dotenv_path), override=False)

_PROD_ENDPOINT = "https://api.thegardens.ai"
_DEV_ENDPOINT = "https://api-dev.thegardens.ai"
_LOCAL_ENDPOINT = "http://localhost:5500"

_DEV_SEARCH_INDEX = "58e4df29-4492-4e7d-9317-b27eba62a911"
_PROD_SEARCH_INDEX = "813d4556-cbd4-4ba9-97f2-a7155f70682f"

_PROD_ECR_REPO = "public.ecr.aws/x2v7f8j4/garden-containers-prod"
_DEV_ECR_REPO = "public.ecr.aws/x2v7f8j4/garden-containers-dev"


class GardenConstants:
    GARDEN_TEST_EMAIL = "garden-test-runner@email.com"
    GARDEN_DIR = os.environ.get("GARDEN_DIR", os.path.expanduser("~/.garden"))
    GARDEN_KEY_STORE = os.path.join(GARDEN_DIR, "tokens.json")
    URL_ENV_VAR_NAME = "GARDEN_MODELS"
    GARDEN_ENDPOINT = (
        _LOCAL_ENDPOINT
        if os.environ.get("GARDEN_ENV") == "local"
        else _DEV_ENDPOINT if os.environ.get("GARDEN_ENV") == "dev" else _PROD_ENDPOINT
    )
    GARDEN_INDEX_UUID = (
        _DEV_SEARCH_INDEX
        if os.environ.get("GARDEN_ENV") in ("dev", "local")
        else _PROD_SEARCH_INDEX
    )
    GARDEN_ECR_REPO = (
        _DEV_ECR_REPO
        if os.environ.get("GARDEN_ENV") in ("dev", "local")
        else _PROD_ECR_REPO
    )

    DEMO_ENDPOINT = "6ed5d749-abc3-4c83-bcad-80837b3d126f"

    # Constants for picking a port to start a Jupyter notebook on
    DEFAULT_JUPYTER_PORT = 9188
    MAX_JUPYTER_PORTS_TO_ATTEMPT = 10

    # The DOIs of entrypoints migrated from DLHub.
    DLHUB_DOIS = set(
        [
            "10.26311/3hz8-as26",
            "10.26311/8s9h-dz64",
            "10.26311/bf7a-7071",
            "10.26311/e2mw-qf63",
            "10.26311/b6zb-ns88",
            "10.26311/q6e2-2p11",
            "10.26311/bkk2-gc19",
            "10.26311/k2bk-hw50",
            "10.26311/bgb7-k519",
            "10.26311/x13g-7f17",
            "10.26311/s8hf-3v65",
            "10.26311/aefd-p769",
            "10.26311/cd31-az33",
        ]
    )

    PREMADE_IMAGES = {
        "3.8-base": "gardenai/base:python-3.8-base",
        "3.9-base": "gardenai/base:python-3.9-base",
        "3.10-base": "gardenai/base:python-3.10-base",
        "3.11-base": "gardenai/base:python-3.11-base",
        "3.8-sklearn": "gardenai/base:python-3.8-sklearn",
        "3.9-sklearn": "gardenai/base:python-3.9-sklearn",
        "3.10-sklearn": "gardenai/base:python-3.10-sklearn",
        "3.11-sklearn": "gardenai/base:python-3.11-sklearn",
        "3.8-tensorflow": "gardenai/base:python-3.8-tensorflow",
        "3.9-tensorflow": "gardenai/base:python-3.9-tensorflow",
        "3.10-tensorflow": "gardenai/base:python-3.10-tensorflow",
        "3.11-tensorflow": "gardenai/base:python-3.11-tensorflow",
        "3.8-torch": "gardenai/base:python-3.8-torch",
        "3.9-torch": "gardenai/base:python-3.9-torch",
        "3.10-torch": "gardenai/base:python-3.10-torch",
        "3.11-torch": "gardenai/base:python-3.11-torch",
        "3.8-all-extras": "gardenai/base:python-3.8-all",
        "3.9-all-extras": "gardenai/base:python-3.9-all",
        "3.10-all-extras": "gardenai/base:python-3.10-all",
        "3.11-all-extras": "gardenai/base:python-3.11-all",
    }

    IMAGES_TO_FLAVOR = {
        "3.8-base": "empty.ipynb",
        "3.9-base": "empty.ipynb",
        "3.10-base": "empty.ipynb",
        "3.11-base": "empty.ipynb",
        "3.8-sklearn": "sklearn.ipynb",
        "3.9-sklearn": "sklearn.ipynb",
        "3.10-sklearn": "sklearn.ipynb",
        "3.11-sklearn": "sklearn.ipynb",
        "3.8-tensorflow": "tensorflow.ipynb",
        "3.9-tensorflow": "tensorflow.ipynb",
        "3.10-tensorflow": "tensorflow.ipynb",
        "3.11-tensorflow": "tensorflow.ipynb",
        "3.8-torch": "torch.ipynb",
        "3.9-torch": "torch.ipynb",
        "3.10-torch": "torch.ipynb",
        "3.11-torch": "torch.ipynb",
        "3.10-all-extras": "empty.ipynb",
    }
