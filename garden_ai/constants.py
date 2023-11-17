import os

_PROD_ENDPOINT = "https://nu3cetwc84.execute-api.us-east-1.amazonaws.com/garden_prod"
_DEV_ENDPOINT = "https://y0ipq1bueb.execute-api.us-east-1.amazonaws.com/garden_dev"

_DEV_SEARCH_INDEX = "58e4df29-4492-4e7d-9317-b27eba62a911"
_PROD_SEARCH_INDEX = "813d4556-cbd4-4ba9-97f2-a7155f70682f"


class GardenConstants:
    GARDEN_TEST_EMAIL = "garden-test-runner@email.com"
    GARDEN_DIR = os.path.expanduser("~/.garden")
    URL_ENV_VAR_NAME = "GARDEN_MODELS"
    GARDEN_ENDPOINT = (
        _PROD_ENDPOINT if os.environ.get("GARDEN_ENV") == "prod" else _DEV_ENDPOINT
    )
    GARDEN_INDEX_UUID = (
        _PROD_SEARCH_INDEX
        if os.environ.get("GARDEN_ENV") == "prod"
        else _DEV_SEARCH_INDEX
    )
    DLHUB_ENDPOINT = "86a47061-f3d9-44f0-90dc-56ddc642c000"

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
