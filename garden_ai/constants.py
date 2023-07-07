import os


class GardenConstants:
    SCAFFOLDED_MODEL_NAME = "YOUR MODEL'S NAME HERE"
    GARDEN_TEST_EMAIL = "garden-test-runner@email.com"
    GARDEN_DIR = os.path.expanduser("~/.garden")
    URL_ENV_VAR_NAME = "GARDEN_MODELS"
    GARDEN_ENDPOINT = os.environ.get(
        "GARDEN_ENDPOINT",
        "https://nu3cetwc84.execute-api.us-east-1.amazonaws.com/garden_prod",
    )
    DLHUB_ENDPOINT = "86a47061-f3d9-44f0-90dc-56ddc642c000"
