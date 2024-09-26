from __future__ import annotations

import inspect
import logging
from functools import wraps
from typing import Any, Callable
from uuid import UUID

import globus_compute_sdk  # type: ignore

from garden_ai.constants import GardenConstants
from garden_ai.model_connectors import ModelConnector, ModelMetadata
from garden_ai.schemas.entrypoint import (
    DatasetMetadata,
    EntrypointMetadata,
    PaperMetadata,
    RegisteredEntrypointMetadata,
    RepositoryMetadata,
)

logger = logging.getLogger()


class Entrypoint:
    """Represents a reproducible, remotely executable function (and its metadata) registered by the Garden service.

    This class is geared towards users wishing to invoke published entrypoints. It defines a `__call__` method for convenience to invoke the function on a remote Globus Compute endpoint, defaulting to the free Garden demo endpoint.

    It is typically created via the client's [get_entrypoint][garden_ai.GardenClient.get_entrypoint] method or accessed as an attribute on a [Garden][garden_ai.Garden].

    Publishers describe and define their entrypoints using the [@entrypoint][garden_ai.entrypoint] decorator in a notebook.

    Attributes:
        metadata (RegisteredEntrypointMetadata): The entrypoint's full metadata. Includes citation metadata as well as information used internally by Garden, such as the Globus Compute function UUID.

    Example:
        ```python
        # Accessed via a published garden
        garden = client.get_garden("my_garden_doi")
        result = garden.my_entrypoint(my_data, endpoint="endpoint_uuid")

        # OR fetched directly:
        my_entrypoint = client.get_entrypoint("my_entrypoint_doi")
        result = my_entrypoint(my_data, endpoint="endpoint_uuid")
        ```
    """  # noqa: E501

    def __init__(self, metadata: RegisteredEntrypointMetadata):
        self.metadata = metadata

    def __call__(
        self,
        *args,
        endpoint: str | UUID = GardenConstants.DEMO_ENDPOINT,
        **kwargs,
    ) -> Any:
        """Remotely execute this entrypoint's function on a specified Globus Compute endpoint.

        This method allows the Entrypoint to be called like a regular function, but it executes the underlying function in a container on a remote endpoint.

        Args:
            *args: Positional arguments passed directly to the entrypoint function.
            endpoint: The Globus Compute endpoint UUID where the function should be executed. Defaults to the free Garden demo endpoint.
            **kwargs: Keyword arguments passed directly to the entrypoint function.

        Returns:
            Any: The result returned by the remotely executed entrypoint function.

        Raises:
            GlobusComputeError: If there's an error in executing the function on the remote endpoint.

        Note:
            The specified remote compute endpoint must support containers in order to run the function. See also: [Globus Compute Endpoint - Containerized Environments](https://globus-compute.readthedocs.io/en/latest/endpoints/single_user.html#containerized-environments).
        """  # noqa: E501
        # delayed import so dill doesn't try to serialize console ref
        from garden_ai.app.console import console

        if self._is_dlhub_entrypoint():
            args = ({"inputs": args[0], "parameters": [], "debug": False},)

        with globus_compute_sdk.Executor(endpoint_id=str(endpoint)) as gce:
            with console.status(
                f"[bold green] executing remotely on endpoint {endpoint}"
            ):
                future = gce.submit_to_registered_function(
                    function_id=str(self.metadata.func_uuid), args=args, kwargs=kwargs
                )
                result = future.result()
                if self._is_dlhub_entrypoint():
                    inner_result = result[0]
                    if inner_result[1]["success"]:
                        return inner_result[0]
                    else:
                        return result
                else:
                    return result

    def _is_dlhub_entrypoint(self) -> bool:
        """
        There are 13 DLHub models that we converted to Garden entrypoints.
        We know their DOIs. We can use this to check if a DOI is a DLHub entrypoint.
        If so, we convert the user's input into the format that DLHub models expect.
        We also just pass along the model output to the user and strip the rest.
        """
        return self.metadata.doi in GardenConstants.DLHUB_DOIS

    def _repr_html_(self) -> str:
        # delayed import so dill doesn't try to serialize tabulate ref
        from tabulate import tabulate

        style = "<style>th {text-align: left;}</style>"
        title = f"<h2>{self.metadata.title}</h2>"
        details = f"<p>Authors: {', '.join(self.metadata.authors)}<br>DOI: {self.metadata.doi}</p>"
        optional = "<h3>Additional data</h3>" + tabulate(
            [
                (field, val)
                for field, val in self.metadata.model_dump(
                    exclude={"title", "authors", "doi", "owner_identity_id", "id"}
                ).items()
                if val
            ],
            tablefmt="html",
        )
        return style + title + details + optional

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.metadata == other.metadata


class EntrypointIdempotencyError(Exception):
    """Raised when an entrypoint function is found to be non-idempotent."""

    pass


def entrypoint(
    metadata: EntrypointMetadata,
    garden_doi: str | None = None,
    model_connectors: list[ModelConnector] | None = None,
    datasets: list[DatasetMetadata] | None = None,
    papers: list[PaperMetadata] | None = None,
    repositories: list[RepositoryMetadata] | None = None,
):
    """Decorator for marking and enriching functions as entrypoints for publication.

    This decorator designates a function as an entrypoint in the Garden system. This prepares the function for registration and remote execution via Globus Compute, enriches it with metadata, and links user-provided metadata of related resources such as datasets or papers.

    Args:
        metadata (EntrypointMetadata): Core metadata describing the entrypoint function.
        garden_doi (str | None): The DOI of the Garden to which this entrypoint should be added. If None, the entrypoint won't be automatically added to any Garden, unless a "notebook-global" DOI was specified (interactively in the top cell of the notebook or with the --doi CLI arg).
        model_connectors (list[ModelConnector] | None): List of ModelConnector objects associated with this entrypoint, used for connecting to and retrieving model artifacts.
        datasets (list[DatasetMetadata] | None): User-provided metadata about datasets used or produced by this entrypoint.
        papers (list[PaperMetadata] | None): User-provided metadata about related research papers or publications.
        repositories (list[RepositoryMetadata] | None): User-provided metadata about related code repositories.

    Notes:
        - This decorator is used within a Jupyter notebook, which can be published with the Garden service.
        - This decorator can be used as many times as desired to publish multiple entrypoints from the same notebook.
        - If garden_doi is provided, the user must be the owner of the garden.
        - If garden_doi is provided, it overrides any "notebook-global" garden DOI for that entrypoint.
        - If model_connectors are provided, their metadata will be extracted and added to the entrypoint's metadata (along with datasets, papers etc)

    Example:
        ```python
        @entrypoint(
            metadata=EntrypointMetadata(title="My Entrypoint Function", authors=["Alice"]),
            garden_doi="10.1234/my-garden",
            model_connectors=[my_model_connector],
            datasets=[my_training_dataset_metatada],
        )
        def my_entrypoint_function(data):
            # Function implementation
            return result
        ```
    """  # noqa: E501

    def decorate(func):
        entrypoint_metadata = metadata.model_copy(deep=True)
        # connect any related metadata
        if datasets:
            entrypoint_metadata.datasets += datasets
        if papers:
            entrypoint_metadata.papers += papers
        if repositories:
            entrypoint_metadata.repositories += repositories
        if model_connectors:
            for connector in model_connectors:
                assert (
                    connector.metadata is not None
                ), "ModelConnector failed to infer metadata"
                model_meta: ModelMetadata = connector.metadata
                entrypoint_metadata.models += [model_meta]

        # set private attributes & attach metadata to function object for
        # the save_session_and_metadata publishing script
        entrypoint_metadata._target_garden_doi = garden_doi
        entrypoint_metadata._function_text = inspect.getsource(func)
        func._entrypoint_metadata = entrypoint_metadata
        return func

    return decorate


def entrypoint_test(entrypoint_func: Callable) -> Callable:
    """Decorator to mark a "test function" for a specific entrypoint.

    This decorator serves two primary purposes:

    1. It associates the test function with its corresponding entrypoint, providing example usage and aiding in documentation.
    2. It prevents test functions from running during the final publication process, allowing users to test their entrypoint interactively from within its notebook without worrying about unintended side effects.

    The decorator also enforces idempotency of the entrypoint function by running the test twice during publication and comparing results.

    Args:
        entrypoint_func (Callable): The entrypoint function that this test is associated with. Must be a function previously decorated with [@entrypoint][garden_ai.entrypoint].

    Returns:
        The decorated test function.

    Raises:
        EntrypointIdempotencyError: When the entrypoint function is found to be non-idempotent, i.e., it produces different results or errors when called twice with the same inputs.
        ValueError: If the provided entrypoint_func is not a valid entrypoint function.

    Example:
        ```python
        @entrypoint(metadata=some_metadata)
        def my_entrypoint(data):
            return process_data(data)

        @entrypoint_test(my_entrypoint)
        def test_my_entrypoint():
            test_data = [1, 2, 3]
            result = my_entrypoint(test_data)
            assert result == [2, 4, 6]
            return result
        ```

    Note:
        The test function should call the entrypoint function with useful representative inputs as an example to help other users invoke it remotely.
    """  # noqa: E501
    if not entrypoint_func or not entrypoint_func._entrypoint_metadata:  # type: ignore
        raise ValueError("Please pass in a valid @entrypoint-decorated function.")

    def decorate(test_func):
        test_function_text = inspect.getsource(test_func)
        entrypoint_func._entrypoint_metadata._test_functions.append(test_function_text)

        @wraps(test_func)
        def inner(*args, **kwargs):
            import os

            # this flag is set during publication time in
            # containers.build_notebook_session_image
            if os.environ.get("GARDEN_SKIP_TESTS") == str(True):
                return None
            else:
                import importlib.util

                # call the test_func once
                result = test_func(*args, **kwargs)

                # Call the test_func again with the same args to enforce idempotency.
                if importlib.util.find_spec("numpy") is not None:
                    import numpy  # type: ignore

                    if isinstance(result, numpy.ndarray):
                        if numpy.array_equal(
                            result, test_func(*args, **kwargs), equal_nan=True
                        ):
                            return result
                        else:
                            raise EntrypointIdempotencyError(
                                "Please ensure your entrypoint can be called more than once without errors."
                            )
                if importlib.util.find_spec("pandas") is not None:
                    import pandas  # type: ignore

                    if isinstance(result, pandas.DataFrame):
                        if result.equals(test_func(*args, **kwargs)):
                            return result
                        else:
                            raise EntrypointIdempotencyError(
                                "Please ensure your entrypoint can be called more than once without errors."
                            )
                if result != test_func(*args, **kwargs):
                    raise EntrypointIdempotencyError(
                        "Please ensure your entrypoint can be called more than once without errors."
                    )
                return result

        return inner

    return decorate
