from __future__ import annotations

import inspect
import logging
from datetime import datetime
from functools import wraps
from typing import Any, List, Optional, Union, Callable
from uuid import UUID

import globus_compute_sdk  # type: ignore
from pydantic import BaseModel, Field, PrivateAttr

from garden_ai.constants import GardenConstants
from garden_ai.datacite import (
    Creator,
    DataciteSchema,
    Description,
    Identifier,
    Subject,
    Title,
    Types,
)
from garden_ai.mlmodel import ModelMetadata
from garden_ai.utils.misc import JSON


logger = logging.getLogger()


class Repository(BaseModel):
    """
    The `Repository` class represents all the metadata we want to \
    publically expose about the repositories used to build the entrypoint.

    Attributes:
        repo_name (str):
            A title that the repository can be referenced by.
        url (str):
            A link where this repository can be publically viewed.
        contributors List[str]:
            Acknowledge contributors to the development of \
            this repository.

    """

    repo_name: str = Field(...)
    url: str = Field(...)
    contributors: List[str] = Field(default_factory=list)


class Paper(BaseModel):
    """
    The `Paper` class represents all the metadata we want to \
    publically expose about the paper used to build the entrypoint.

    Attributes:
        title (str):
            The official title that the paper can be referenced by.
        authors List[str]:
            The main researchers involved in producing the paper. Personal name \
            format should be: "Family, Given". Order is preserved. (at least one required)
        doi (str):
            The digital object identifier of the paper. (Optional)
        citation (str):
            Description of how the paper may be cited officially. (Optional)

    """

    title: str = Field(...)
    authors: List[str] = Field(default_factory=list)
    doi: Optional[str] = Field(None)
    citation: Optional[str] = Field(None)


class Step(BaseModel):
    """
    The `Step` class represents a key function in an entrypoint that a publisher wants to highlight.

    Attributes:
        function_name (str):
            The name of the step function.
        function_text (str):
            The full Python code that makes up the function.
        description (str):
            An optional string describing the function.
    """

    function_name: str = Field(...)
    function_text: str = Field(...)
    description: Optional[str] = Field(None)


class EntrypointMetadata(BaseModel):
    """Metadata for an entrypoint prior to its registration. \
    Passed to the `garden_entrypoint` decorator during the registration process.

    Use the optional doi field if you have already registered this entrypoint \
    and want to update it under the same DOI.

    Attributes:
        doi: Optional. If you have a DOI you want to use for this entrypoint, specify it here. \
        (Especially if you have already registered an entrypoint and are updating it.) \
        Otherwise we will generate a DOI for you.
        title: A short title that describes the entrypoint.
        description: A longer free text description of this entrypoint.
        authors: A list of the authors of this entrypoint. You need at least one.
        short_name: This will be the name of the Python method that people call to invoke your entrypoint.
        year: When did you make this entrypoint? (Defaults to current year)
        tags: Helpful tags
        repositories: List of related code repositories, like GitHub or GitLab repos.
        papers: List of related papers, like a paper that describes the model you are publishing here.
    """

    doi: Optional[str] = Field(None)
    title: str = Field(...)
    authors: List[str] = Field(...)
    short_name: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    year: str = Field(default_factory=lambda: str(datetime.now().year))
    tags: List[str] = Field(default_factory=list, unique_items=True)
    models: List[ModelMetadata] = Field(default_factory=list)
    repositories: List[Repository] = Field(default_factory=list)
    papers: List[Paper] = Field(default_factory=list)
    # The PrivateAttrs below are used internally for publishing.
    _test_functions: List[str] = PrivateAttr(default_factory=list)
    _target_garden_doi: Optional[str] = PrivateAttr(None)
    _as_step: Optional[Step] = PrivateAttr(None)


class RegisteredEntrypoint(EntrypointMetadata):
    """Metadata of a completed and registered entrypoint.
    Can be added to a Garden and executed on a remote Globus Compute endpoint.

    Has all the user-given metadata from EntrypointMetadata plus extra fields added by Garden
    during publication.

    Attributes:
        func_uuid: The ID of the Globus Compute function registered for this entrypoint.
        container_uuid: ID returned from Globus Compute's register_container.
        base_image_uri: location of the base image used by this entrypoint. eg docker://index.docker.io/maxtuecke/garden-ai:python-3.9-jupyter
        full_image_uri: The name and location of the complete image used by this entrypoint.
        notebook_url: Link to the notebook used to build this entrypoint.
        steps: Ordered list of Python functions that the entrypoint author wants to highlight.
        test_functions: List of test functions that exercise the entrypoint.
    """

    doi: str = Field(
        ...
    )  # Repeating this field from base class because DOI is mandatory for RegisteredEntrypoint
    func_uuid: UUID = Field(...)
    container_uuid: UUID = Field(...)
    base_image_uri: Optional[str] = Field(None)
    full_image_uri: Optional[str] = Field(None)
    notebook_url: Optional[str] = Field(None)
    steps: List[Step] = Field(default_factory=list)
    test_functions: List[str] = Field(default_factory=list)

    def __call__(
        self,
        *args: Any,
        endpoint: Union[UUID, str] = None,
        **kwargs: Any,
    ) -> Any:
        """Remotely execute this ``RegisteredEntrypoint``'s function from the function uuid.

        Args:
            *args (Any):
                Input data passed through the first step in the entrypoint
            endpoint (UUID | str | None):
                Where to run the entrypoint. Must be a valid Globus Compute endpoint UUID.
                If no endpoint is specified, the DLHub default compute endpoint is used.
            **kwargs (Any):
                Additional keyword arguments passed directly to the first step in the entrypoint.

        Returns:
            Results from the entrypoint's composed steps called with the given input data.
        """
        # delayed import so dill doesn't try to serialize console ref
        from garden_ai.app.console import console

        if not endpoint:
            endpoint = GardenConstants.DLHUB_ENDPOINT

        with globus_compute_sdk.Executor(endpoint_id=str(endpoint)) as gce:
            with console.status(
                f"[bold green] executing remotely on endpoint {endpoint}"
            ):
                future = gce.submit_to_registered_function(
                    function_id=str(self.func_uuid), args=args, kwargs=kwargs
                )
                return future.result()

    def _repr_html_(self) -> str:
        # delayed import so dill doesn't try to serialize tabulate ref
        from tabulate import tabulate

        style = "<style>th {text-align: left;}</style>"
        title = f"<h2>{self.title}</h2>"
        details = f"<p>Authors: {', '.join(self.authors)}<br>DOI: {self.doi}</p>"
        optional = "<h3>Additional data</h3>" + tabulate(
            [
                (field, val)
                for field, val in self.dict().items()
                if field not in ("title", "authors", "doi", "steps") and val
            ],
            tablefmt="html",
        )
        return style + title + details + optional

    def datacite_json(self) -> JSON:
        """Parse this `Entrypoint`'s metadata into a DataCite-schema-compliant JSON string."""

        # Leverages a pydantic class `DataCiteSchema`, which was automatically generated from:
        # https://github.com/datacite/schema/blob/master/source/json/kernel-4.3/datacite_4.3_schema.json
        #
        # The JSON returned by this method would be the "attributes" part of a DataCite request body.

        return DataciteSchema(
            identifiers=[Identifier(identifier=self.doi, identifierType="DOI")],
            types=Types(resourceType="AI/ML Entrypoint", resourceTypeGeneral="Software"),  # type: ignore
            creators=[Creator(name=name) for name in self.authors],
            titles=[Title(title=self.title)],
            publisher="thegardens.ai",
            publicationYear=self.year,
            subjects=[Subject(subject=tag) for tag in self.tags],
            descriptions=(
                [
                    Description(description=self.description, descriptionType="Other")  # type: ignore
                ]
                if self.description
                else None
            ),
        ).json()


def garden_entrypoint(
    metadata: EntrypointMetadata,
    garden_doi: str = None,
    model_connectors=None,
):
    def decorate(func):
        if model_connectors:
            metadata.models += [connector.metadata for connector in model_connectors]
        metadata._as_step = Step(
            function_text=inspect.getsource(func),
            function_name=func.__name__,
            description="Top level entrypoint function",
        )
        metadata._target_garden_doi = garden_doi
        # let func carry its own metadata
        func._garden_entrypoint = metadata
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
    """
    if not entrypoint_func or not entrypoint_func._garden_entrypoint:  # type: ignore
        raise ValueError("Please pass in a valid entrypoint function")

    def decorate(test_func):
        test_function_text = inspect.getsource(test_func)
        entrypoint_func._garden_entrypoint._test_functions.append(test_function_text)

        @wraps(test_func)
        def inner(*args, **kwargs):
            import os

            # this flag is set during publication time in
            # containers.build_notebook_session_image
            if os.environ.get("GARDEN_SKIP_TESTS") == str(True):
                return None
            else:
                return test_func(*args, **kwargs)

        return inner

    return decorate


def garden_step(description: str = None):
    def decorate(func):
        func._garden_step = Step(
            function_text=inspect.getsource(func),
            function_name=func.__name__,
            description=description,
        )
        return func

    return decorate
