import pickle
import mlflow


class ModelUploadException(Exception):
    """Exception raised when an attempt to upload a model to ML Flow fails"""
    pass


def upload_model(model_path: str, model_name: str, user_email: str, extra_pip_requirements: list[str] = None):
    full_model_name = f'{user_email}/{model_name}'
    try:
        f = open(model_path, 'rb')
    except IOError as e:
        raise ModelUploadException('Could not open file ' + model_path) from e
    try:
        loaded_model = pickle.load(f)
    except pickle.PickleError as e:
        raise ModelUploadException('Could not parse model at ' + model_path) from e
    try:
        mlflow.sklearn.log_model(
            loaded_model,
            user_email,
            registered_model_name=full_model_name,
            extra_pip_requirements=extra_pip_requirements
        )
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(full_model_name, stages=['None'])
    except mlflow.MlflowException as e:
        raise ModelUploadException('Could not upload model.') from e
    if len(versions) == 0:
        raise ModelUploadException('Could not retrieve model version.')
    version_number = versions[0].version

    return f'{full_model_name}/{version_number}'