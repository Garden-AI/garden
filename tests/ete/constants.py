import os
import uuid
from garden_ai import GardenConstants


class ETEConstants:
    def __init__(self):
        self.mock_container_cache = {
            "3899656743": "b9cf409f-d5f2-4956-b198-0d90ffa133e6",  # sklearn py3.8.16
            "2159326884": "946155fa-c79e-48d1-9316-42353d4f97c3",  # sklearn py3.9.17
            "802392313": "7705f553-cc5c-4404-8e4c-a37cf718571e",  # sklearn py3.10.12
            "3723963236": "e6beb8d0-fef6-470d-ae8d-74e9bcbffe10",  # tf py3.8.16
            "789531598": "58f51be9-ef56-4064-add1-f030f59da6aa",  # tf py3.9.17
            "4124804071": "c5639554-e46d-444d-adcb-3dbc1e4d6ab8",  # tf py3.10.12
            "3292203339": "63c2caec-7eb2-4930-b778-d74e13351bf6",  # torch py3.8.16
            "2902439368": "93f0d0c2-ea65-41c1-8ee8-bb6713fbec59",  # torch py3.9.17
            "60328853": "846c5432-eed1-41bc-b469-ef7974b6598c",  # torch py3.10.12
        }

        self.fresh_endpoint_name = "Garden-ETE-Test-Endpoint"

        self.key_store_path = GardenConstants.GARDEN_DIR

        self.garden_title = f"ETE-Test-Garden-{str(uuid.uuid4())}"

        self.scaffolded_entrypoint_folder_name = "ete_test_entrypoint_title"
        self.entrypoint_template_name = "ete_entrypoint_cc"

        self.sklearn_entrypoint_path = os.path.join(
            self.key_store_path, "sklearn_entrypoint.py"
        )
        self.sklearn_entrypoint_name = "ETESklearnEntrypoint"
        self.sklearn_model_name = "ETE-Test-Model-Sklearn"

        self.sklearn_pre_entrypoint_path = os.path.join(
            self.key_store_path, "sklearn_pre_entrypoint.py"
        )
        self.sklearn_pre_entrypoint_name = "ETESklearnPreEntrypoint"
        self.sklearn_pre_model_name = "ETE-Test-Model-Sklearn-Pre"

        self.tf_entrypoint_path = os.path.join(self.key_store_path, "tf_entrypoint.py")
        self.tf_entrypoint_name = "ETETfEntrypoint"
        self.tf_model_name = "ETE-Test-Model-Tf"

        self.torch_entrypoint_path = os.path.join(
            self.key_store_path, "torch_entrypoint.py"
        )
        self.torch_entrypoint_name = "ETETorchEntrypoint"
        self.torch_model_name = "ETE-Test-Model-Torch"

        self.sklearn_model_location = os.path.abspath("./models/sklearn_model.pkl")
        self.sklearn_pre_model_location = os.path.abspath(
            "./models/sklearn_preprocessor_model.pkl"
        )
        self.tf_model_location = os.path.abspath("./models/keras_model.keras")
        self.torch_model_location = os.path.abspath("./models/torch_model.pth")

        self.sklearn_input_data_location = os.path.abspath(
            "./models/sklearn_test_input.pkl"
        )
        self.sklearn_pre_input_data_location = os.path.abspath(
            "./models/sklearn_preprocessor_test_input.pkl"
        )
        self.tf_input_data_location = os.path.abspath("./models/keras_test_input.pkl")
        self.torch_input_data_location = os.path.abspath(
            "./models/torch_test_input.pkl"
        )

        self.sklearn_expected_data_location = os.path.abspath(
            "./models/sklearn_test_expected.pkl"
        )
        self.sklearn_pre_expected_data_location = os.path.abspath(
            "./models/sklearn_preprocessor_test_expected.pkl"
        )
        self.tf_expected_data_location = os.path.abspath(
            "./models/keras_test_expected.pkl"
        )
        self.torch_expected_data_location = os.path.abspath(
            "./models/torch_test_expected.pkl"
        )

        self.sklearn_model_reqs_location = os.path.abspath(
            "./models/sklearn_requirements.txt"
        )
        self.tf_model_reqs_location = os.path.abspath("./models/keras_requirements.txt")
        self.torch_model_reqs_location = os.path.abspath(
            "./models/torch_requirements.txt"
        )

        self.sklearn_func = "predict"
        self.sklearn_pre_func = "transform"
        self.tf_func = "predict"
        self.torch_func = "predict"
        self.custom_func = "predict"

        self.entrypoint_template_location = os.path.abspath("./templates")

        self.example_garden_data = {
            "authors": ["Test Garden Author"],
            "title": "ETE Test Garden Title",
            "contributors": ["Test Garden Contributor"],
            "year": "2023",
            "description": "ETE Test Garden Description",
        }

        self.example_entrypoint_data = {
            "authors": ["Test Entrypoint Author"],
            "title": "ETE Test Entrypoint Title",
            "contributors": ["Test Entrypoint Contributor"],
            "year": "2023",
            "description": "ETE Test Entrypoint Description",
        }

        self.custom_model_location = None
        self.custom_model_flavor = None
        self.custom_model_reqs = None

        self.custom_model_name = "ETE-Test-Model-Custom"
        self.custom_entrypoint_name = "ETECustomEntrypoint"
        self.custom_entrypoint_path = os.path.join(
            self.key_store_path, "custom_entrypoint.py"
        )
        self.custom_make_new_entrypoint = True

        self.local_files_list = [
            self.sklearn_entrypoint_path,
            self.tf_entrypoint_path,
            self.torch_entrypoint_path,
            self.sklearn_pre_entrypoint_path,
            self.custom_entrypoint_path,
            os.path.join(self.key_store_path, self.scaffolded_entrypoint_folder_name),
            os.path.join(self.key_store_path, "data.json"),
            os.path.join(self.key_store_path, "tokens.json"),
        ]

        self.default_endpoint = GardenConstants.DLHUB_ENDPOINT

        self.custom_serialize = None
