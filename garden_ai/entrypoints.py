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
    def __init__(self, metadata: RegisteredEntrypointMetadata):
        self.metadata = metadata

    def __call__(
        self,
        *args,
        endpoint: str | UUID | None = GardenConstants.DEMO_ENDPOINT,
        **kwargs,
    ) -> Any:
        """Remotely execute this entrypoint's function from its function uuid.

        Args:
            *args (Any):
                Input data passed directly to the entrypoint function.
            endpoint (UUID | str | None):
                Where to run the entrypoint. Must be a valid Globus Compute endpoint UUID.
                If no endpoint is specified, the default demo endpoint is used.
            **kwargs (Any):
                Additional keyword arguments passed directly to the entrypoint function.

        Returns:
            Results from invoking the entrypoint function with the given input data.
        """
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


def garden_entrypoint(
    metadata: EntrypointMetadata,
    garden_doi: str | None = None,
    model_connectors: list[ModelConnector] | None = None,
    datasets: list[DatasetMetadata] | None = None,
    papers: list[PaperMetadata] | None = None,
    repositories: list[RepositoryMetadata] | None = None,
):
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
                model_meta: ModelMetadata = connector.metadata
                entrypoint_metadata.models += [model_meta]

        # set private attributes & attach metadata to function object for
        # the save_session_and_metadata publishing script
        entrypoint_metadata._target_garden_doi = garden_doi
        entrypoint_metadata._function_text = inspect.getsource(func)
        func._entrypoint_metadata = entrypoint_metadata
        return func

    return decorate


def entrypoint_test(entrypoint_func: Callable):
    """Mark a function as a 'test function' of an entrypoint.

    Marked test functions won't run at publication time, so they can be safely
    called at the top-level of a notebook without causing unintended side-effects.

    Example:

        ```python
        @garden_entrypoint(...)
        def my_entrypoint(*args, **kwargs):
            ...

        @entrypoint_test(my_entrypoint)
        def test_my_entrypoint():
            ...
            results = my_entrypoint(...)
            ...
            return results

        ```

    Raises:
        EntrypointIdempotencyError: When entrypoint_func is found to be non-idempotent, i.e. It cannot be called twice
            without errors.
    """
    if not entrypoint_func or not entrypoint_func._entrypoint_metadata:  # type: ignore
        raise ValueError("Please pass in a valid entrypoint function")

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
