import os

_garden_dir = os.path.expanduser("~/.garden")


class GardenConstants:
    SCAFFOLDED_MODEL_NAME = "YOUR MODEL'S NAME HERE"
    GARDEN_TEST_EMAIL = "garden-test-runner@email.com"
    GARDEN_DIR = os.path.expanduser("~/.garden")
    GARDEN_ENDPOINT = os.environ.get(
        "GARDEN_ENDPOINT",
        "https://nu3cetwc84.execute-api.us-east-1.amazonaws.com/garden_prod",
    )
