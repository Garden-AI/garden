class _Model:
    def __init__(
        self,
        model_full_name: str,
    ):
        self.model = None
        self.full_name = model_full_name

        return

    def download_and_stage(
        self, presigned_download_url: str, full_model_name: str
    ) -> str:
        import os
        import pathlib
        import zipfile

        import requests

        try:
            # Not taking MODEL_STAGING_DIR from constants
            # so that this class has no intra-garden dependencies
            staging_dir = pathlib.Path(os.path.expanduser("~/.garden")) / "mlflow"
            download_dir = staging_dir / full_model_name
            download_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as pe:
            raise PermissionError("Could not create model staging directory") from pe

        zip_filepath = str(download_dir / "model.zip")

        try:
            response = requests.get(presigned_download_url, stream=True)
            response.raise_for_status()
        except requests.RequestException as re:
            raise Exception(
                f"Could not download model from presigned url. URL: {presigned_download_url}"
            ) from re

        try:
            with open(zip_filepath, "wb") as f:
                f.write(response.content)
        except IOError as ioe:
            raise IOError(
                f"Failed to write model to disk at location {zip_filepath}."
            ) from ioe

        extraction_dir = download_dir / "unzipped"
        unzipped_path = str(download_dir / extraction_dir)

        try:
            with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
                zip_ref.extractall(unzipped_path)
        except (FileNotFoundError, zipfile.BadZipFile) as fe:
            raise Exception(
                "Failed to unzip model directory from Garden model repository."
            ) from fe

        return unzipped_path

    @staticmethod
    def get_download_url(full_model_name: str) -> str:
        import json
        import os

        model_url_json = os.environ.get("GARDEN_MODELS", None)
        if not model_url_json:
            raise KeyError(
                "GARDEN_MODELS environment variable was not set. Cannot download model."
            )
        try:
            model_url_dict = json.loads(model_url_json)
            return model_url_dict[full_model_name]
        except (json.JSONDecodeError, KeyError) as e:
            raise KeyError(
                f"Could not find url for model {full_model_name} in GARDEN_MODELS env var contents {model_url_json}"
            ) from e

    # Duplicated in this class so that _Model is self-contained
    @staticmethod
    def clear_mlflow_staging_directory():
        import os
        import pathlib
        import shutil

        staging_dir = pathlib.Path(os.path.expanduser("~/.garden")) / "mlflow"
        path = str(staging_dir)
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

    def _lazy_load_model(self):
        """download and deserialize the underlying model, if necessary."""
        import mlflow  # type: ignore

        if self.model is None:
            download_url = self.get_download_url(self.full_name)
            local_model_path = self.download_and_stage(download_url, self.full_name)
            self.model = mlflow.pyfunc.load_model(
                local_model_path, suppress_warnings=True
            )
            try:
                self.clear_mlflow_staging_directory()
            except Exception as e:
                raise Exception(
                    f"Could not clean up model staging directory. Check permissions on {self.staging_dir}"
                ) from e
        return

    def predict(self, data):
        """Generate model predictions.

        The underlying model will be downloaded if it hasn't already.

        input data is passed directly to the underlying model via its respective
        ``predict`` method.

        Parameters
        ----------
        data : pd.DataFrame | pd.Series | np.ndarray | List[Any] | Dict[str, Any]
            Input data fed to the model

        Returns
        --------
        Results of model prediction

        """
        self._lazy_load_model()
        return self.model.predict(data)
