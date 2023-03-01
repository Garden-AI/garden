from functools import lru_cache
import mlflow.pyfunc  # type: ignore


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
    return mlflow.pyfunc.load_model(
        model_uri=model_uri, suppress_warnings=False, dst_path=None
    )
