import pickle
from functools import lru_cache
from typing import List

import mlflow  # type: ignore
from mlflow.pyfunc import load_model  # type: ignore


class ModelUploadException(Exception):
    """Exception raised when an attempt to upload a model to ML Flow fails"""

    pass


def upload_model(
    model_path: str,
    model_name: str,
    user_email: str,
    extra_pip_requirements: List[str] = None,
) -> str:
    full_model_name = f"{user_email}-{model_name}"
    try:
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
            mlflow.sklearn.log_model(
                loaded_model,
                user_email,
                registered_model_name=full_model_name,
                extra_pip_requirements=extra_pip_requirements,
            )
    except IOError as e:
        raise ModelUploadException("Could not open file " + model_path) from e
    except pickle.PickleError as e:
        raise ModelUploadException("Could not parse model at " + model_path) from e
    except mlflow.MlflowException as e:
        raise ModelUploadException("Could not upload model.") from e

    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(full_model_name, stages=["None"])
    if len(versions) == 0:
        raise ModelUploadException("Could not retrieve model version.")
    version_number = versions[0].version
    return f"{full_model_name}/{version_number}"


@lru_cache
def Model(model_uri: str) -> mlflow.pyfunc.PyFuncModel:
    """Load a registered model from Garden-AI's (MLflow) tracking server.

    Tip: for large models, using this as a "default argument" in a ``@step``-decorated
    function will trigger the download as soon as the pipeline is initialized,
    _before_ any calls to the pipeline. (Usage elsewhere should be fine, but
    the pipeline won't be able to download the model until is actually called)

    Example:
    --------
    ```python
    import garden_ai
    from garden_ai import step
    ....
    # OK for preceding step to return only a DataFrame
    @step
    def run_inference(
        my_data: pd.DataFrame,
        my_model = garden_ai.Model("models:/..."),  # NOTE: only downloads once when
                                                    # python evaluates the default
    ) -> MyResultType:
    '''Run inference on DataFrame `my_data`, returned by previous step.'''

        result = my_model.predict(my_data)
        return result
    ```

    Notes:
    ------
    This is currently just a wrapper around `mlflow.pyfunc.load_model`. In the future
    this might implement smarter caching behavior, but for now the preferred usage is
    to use this function as a default value for some keyword argument.
    """
    return load_model(model_uri=model_uri, suppress_warnings=False, dst_path=None)
