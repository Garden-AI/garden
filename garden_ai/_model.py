class _Model:
    def __init__(
        self,
        model_full_name: str,
    ):
        self._model = None
        self.full_name = model_full_name

        return

    @property
    def model(self):
        self._lazy_load_model()
        return self._model

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
        import yaml

        if self._model is None:
            download_url = self.get_download_url(self.full_name)
            local_model_path = self.download_and_stage(download_url, self.full_name)
            mlflow_yaml_path = local_model_path + "/MLmodel"
            try:
                with open(mlflow_yaml_path, "r") as stream:
                    mlflow_metadata = yaml.safe_load(stream)
                mlflow_load_strategy = mlflow_metadata["metadata"][
                    "garden_load_strategy"
                ]
            except KeyError:
                # Default to mlflow.pyfunc if can't find garden_load_strategy
                mlflow_load_strategy = "pyfunc"

            if mlflow_load_strategy == "pyfunc":
                self._model = mlflow.pyfunc.load_model(
                    local_model_path, suppress_warnings=True
                )
            elif mlflow_load_strategy == "sklearn":
                self._model = mlflow.sklearn.load_model(local_model_path)
            elif mlflow_load_strategy == "pytorch":
                # Load torch models with mlflow.torch so they can taketorch.tensors inputs
                # Will also cause torch models to fail with np.ndarrays or pd.dataframes inputs
                # Must load instead with mlflow.pyfunc to handel the latter two.
                # TODO add signatures option to allow users to specify input data type to load models accordingly.
                self._model = self._TorchWrapper(
                    mlflow.pytorch.load_model(local_model_path)
                )
            else:
                raise Exception(
                    f"Invlaid garden_load_strategy given: {mlflow_load_strategy}"
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
        data : pd.DataFrame | pd.Series | np.ndarray | List[Any] | Dict[str, Any] | torch.tensor
            Input data fed to the model

        Returns
        --------
        Results of model prediction

        """
        return self.model.predict(data)

    # Dill serialization breaks using new __getattr__ if missing this.
    def __getstate__(self):
        return self.__dict__

    def __getattr__(self, attr):
        if attr == "_model":
            # Dill deserialization infinitely recursive without this
            raise AttributeError()
        if self._model is not None:
            if attr in self.__dict__:
                # Use self definition of attr
                return self.__dict__[attr]
            else:
                # self does not have definition of attr
                # Search _model instead
                return self._model.__getattribute__(attr)
        else:
            # _model is None, lazy load and try again.
            self._lazy_load_model()
            return getattr(self, attr)

    class _TorchWrapper(object):
        """
        Wrapper class for pytorch models.
        Adds predict method to torch models.
        """

        def __init__(self, model):
            self._wrapped_model = model

        def predict(self, data):
            return self._wrapped_model(data)

        def __getattr__(self, attr):
            if attr in self.__dict__:
                # use _TorchWrapper definition of attr
                return getattr(self, attr)
            # _TorchWrapper does not have attr,
            # use _wrapped_model definition of attr instead
            return getattr(self._wrapped_model, attr)
