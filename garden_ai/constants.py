import os
from dotenv import load_dotenv
import pathlib

# get env file from project root
dotenv_path = pathlib.Path(f"{__file__}/../..").resolve() / ".env.shared"
load_dotenv(str(dotenv_path), override=False)

_PROD_ENDPOINT = "https://nu3cetwc84.execute-api.us-east-1.amazonaws.com/garden_prod"
_DEV_ENDPOINT = "https://y0ipq1bueb.execute-api.us-east-1.amazonaws.com/garden_dev"

_DEV_SEARCH_INDEX = "58e4df29-4492-4e7d-9317-b27eba62a911"
_PROD_SEARCH_INDEX = "813d4556-cbd4-4ba9-97f2-a7155f70682f"

_PROD_ECR_REPO = "public.ecr.aws/x2v7f8j4/garden-containers-prod"
_DEV_ECR_REPO = "public.ecr.aws/x2v7f8j4/garden-containers-dev"


class GardenConstants:
    GARDEN_TEST_EMAIL = "garden-test-runner@email.com"
    GARDEN_DIR = os.path.expanduser("~/.garden")
    GARDEN_KEY_STORE = os.path.join(GARDEN_DIR, "tokens.json")
    URL_ENV_VAR_NAME = "GARDEN_MODELS"
    GARDEN_ENDPOINT = (
        _DEV_ENDPOINT if os.environ.get("GARDEN_ENV") == "dev" else _PROD_ENDPOINT
    )
    GARDEN_INDEX_UUID = (
        _DEV_SEARCH_INDEX
        if os.environ.get("GARDEN_ENV") == "dev"
        else _PROD_SEARCH_INDEX
    )
    GARDEN_ECR_REPO = (
        _DEV_ECR_REPO if os.environ.get("GARDEN_ENV") == "dev" else _PROD_ECR_REPO
    )

    DEMO_ENDPOINT = "86a47061-f3d9-44f0-90dc-56ddc642c000"

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
        "3.8-base": "gardenai/base:python-3.8-jupyter",
        "3.9-base": "gardenai/base:python-3.9-jupyter",
        "3.10-base": "gardenai/base:python-3.10-jupyter",
        "3.11-base": "gardenai/base:python-3.11-jupyter",
        "3.8-sklearn": "gardenai/base:python-3.8-jupyter-sklearn",
        "3.9-sklearn": "gardenai/base:python-3.9-jupyter-sklearn",
        "3.10-sklearn": "gardenai/base:python-3.10-jupyter-sklearn",
        "3.11-sklearn": "gardenai/base:python-3.11-jupyter-sklearn",
        "3.8-tensorflow": "gardenai/base:python-3.8-jupyter-tf",
        "3.9-tensorflow": "gardenai/base:python-3.9-jupyter-tf",
        "3.10-tensorflow": "gardenai/base:python-3.10-jupyter-tf",
        "3.11-tensorflow": "gardenai/base:python-3.11-jupyter-tf",
        "3.8-torch": "gardenai/base:python-3.8-jupyter-torch",
        "3.9-torch": "gardenai/base:python-3.9-jupyter-torch",
        "3.10-torch": "gardenai/base:python-3.10-jupyter-torch",
        "3.11-torch": "gardenai/base:python-3.11-jupyter-torch",
        "3.10-all-extras": "gardenai/base:python-3.10-jupyter-all",
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
