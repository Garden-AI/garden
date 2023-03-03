import pickle
import mlflow  # type: ignore
from typing import List


class ModelUploadException(Exception):
    """Exception raised when an attempt to upload a model to ML Flow fails"""
    pass


def upload_model(model_path: str, model_name: str, user_email: str, extra_pip_requirements: List[str] = None):
    full_model_name = f'{user_email}-{model_name}'
    try:
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
            mlflow.sklearn.log_model(
                loaded_model,
                user_email,
                registered_model_name=full_model_name,
                extra_pip_requirements=extra_pip_requirements
            )
    except IOError as e:
        raise ModelUploadException('Could not open file ' + model_path) from e
    except pickle.PickleError as e:
        raise ModelUploadException('Could not parse model at ' + model_path) from e
    except mlflow.MlflowException as e:
        raise ModelUploadException('Could not upload model.') from e

    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(full_model_name, stages=['None'])
    if len(versions) == 0:
        raise ModelUploadException('Could not retrieve model version.')
    version_number = versions[0].version
    return f'{full_model_name}/{version_number}'
