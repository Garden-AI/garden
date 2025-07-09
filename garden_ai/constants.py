import os
from dotenv import load_dotenv
import pathlib

# get env file from project root
dotenv_path = pathlib.Path(f"{__file__}/../..").resolve() / ".env.shared"
load_dotenv(str(dotenv_path), override=False)

_PROD_ENDPOINT = "https://api.thegardens.ai"
_DEV_ENDPOINT = "https://api-dev.thegardens.ai"
_LOCAL_ENDPOINT = "http://localhost:5500"


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

    MIXPANEL_TOKEN = "db71da1b071c8acd33af84921ce88084"

    BACKEND_POLL_INTERVAL_SECONDS: float = float(
        os.environ.get("BACKEND_POLL_INTERVAL_SECONDS", 0.1)
    )
