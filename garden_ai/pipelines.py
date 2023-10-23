from __future__ import annotations

import logging

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import globus_compute_sdk  # type: ignore
from pydantic import BaseModel, Field
from tabulate import tabulate

from garden_ai.app.console import console
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
    publically expose about the repositories used to build the pipeline.

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
    publically expose about the paper used to build the pipeline.

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


class PipelineMetadata(BaseModel):
    """Mere metadata for a pipeline prior to its registration.
    Passed to the `garden_pipeline` decorator during the registration
    process.

    The optional doi field allows one to maintain the same DOI across
    versions of the same pipeline.

    `PipelineMetadata` objects can be described completely by JSON.

    Attributes:
        doi:
        title:
        authors:
        short_name:
        description:
        year:
        tags:
        model_full_names:
        repositories:
        papers:
    """

    doi: Optional[str] = Field(None)
    title: str = Field(...)
    authors: List[str] = Field(...)
    short_name: str = Field(...)
    description: Optional[str] = Field(None)
    year: str = Field(default_factory=lambda: str(datetime.now().year))
    tags: List[str] = Field(default_factory=list, unique_items=True)
    model_full_names: List[str] = Field(default_factory=list)
    repositories: List[Repository] = Field(default_factory=list)
    papers: List[Paper] = Field(default_factory=list)


class RegisteredPipeline(BaseModel):
    """Metadata of a completed and registered pipeline.
    Can be added to a Garden and executed on a remote Globus Compute endpoint.

    `RegisteredPipeline` objects can be described completely by JSON.

    Attributes:
        doi:
        func_uuid:
        container_uuid:
        title:
        authors:
        short_name:
        description:
        year:
        tags:
        model_full_names:
        repositories:
        papers:
    """

    doi: str = Field(...)
    func_uuid: UUID = Field(...)
    container_uuid: UUID = Field(...)
    title: str = Field(...)
    authors: List[str] = Field(...)
    short_name: str = Field(...)
    description: Optional[str] = Field(None)
    year: str = Field(default_factory=lambda: str(datetime.now().year))
    tags: List[str] = Field(default_factory=list, unique_items=True)
    model_full_names: List[str] = Field(default_factory=list)
    repositories: List[Repository] = Field(default_factory=list)
    papers: List[Paper] = Field(default_factory=list)

    def __call__(
        self,
        *args: Any,
        endpoint: Union[UUID, str] = None,
        **kwargs: Any,
    ) -> Any:
        """Remotely execute this ``RegisteredPipeline``'s function from the function uuid.

        Args:
            *args (Any):
                Input data passed through the first step in the pipeline
            endpoint (UUID | str | None):
                Where to run the pipeline. Must be a valid Globus Compute endpoint UUID.
                If no endpoint is specified, the DLHub default compute endpoint is used.
            **kwargs (Any):
                Additional keyword arguments passed directly to the first step in the pipeline.

        Returns:
            Results from the pipeline's composed steps called with the given input data.


        """
        if not endpoint:
            endpoint = GardenConstants.DLHUB_ENDPOINT

        with globus_compute_sdk.Executor(endpoint_id=str(endpoint)) as gce:
            # TODO: refactor below once the remote-calling interface is settled.
            # console/spinner is good ux but shouldn't live this deep in the
            # sdk.
            with console.status(
                f"[bold green] executing remotely on endpoint {endpoint}"
            ):
                future = gce.submit_to_registered_function(
                    function_id=str(self.func_uuid), args=args, kwargs=kwargs
                )
                return future.result()

    def _repr_html_(self) -> str:
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

    def collect_models(self) -> List[ModelMetadata]:
        """Collect the RegisteredModel objects that are present in the local DB corresponding to this Pipeline's list of `model_full_names`."""
        from .local_data import get_local_model_by_name

        models = []
        for model_name in self.model_full_names:
            model = get_local_model_by_name(model_name)
            if model:
                models += [model]
            else:
                logger.warning(
                    f"No record in local database for model {model_name}. "
                    "Published garden will not have detailed metadata for that model."
                )
        return models

    def expanded_metadata(self) -> Dict[str, Any]:
        """Helper: build the "complete" metadata dict with nested ``Model`` metadata.

        Notes:
            When serializing normally with ``registered_pipeline.{dict(), \
            json()}``, only the uris of the models in the pipeline are included. \
            This returns a superset of `registered_pipeline.dict()`, so that the \
            following holds: \
                `pipeline == Registered_Pipeline(**pipeline.expanded_metadata()) == Registered_Pipeline(**pipeline.dict())`

        Returns:
            ``RegisteredPipeline`` metadata dict augmented with a list of ``RegisteredModel`` metadata
        """
        data = self.dict()
        data["models"] = [m.dict() for m in self.collect_models()]
        return data

    def datacite_json(self) -> JSON:
        """Parse this `Pipeline`'s metadata into a DataCite-schema-compliant JSON string."""

        # Leverages a pydantic class `DataCiteSchema`, which was automatically generated from:
        # https://github.com/datacite/schema/blob/master/source/json/kernel-4.3/datacite_4.3_schema.json
        #
        # The JSON returned by this method would be the "attributes" part of a DataCite request body.

        return DataciteSchema(
            identifiers=[Identifier(identifier=self.doi, identifierType="DOI")],
            types=Types(resourceType="AI/ML Pipeline", resourceTypeGeneral="Software"),  # type: ignore
            creators=[Creator(name=name) for name in self.authors],
            titles=[Title(title=self.title)],
            publisher="thegardens.ai",
            publicationYear=self.year,
            subjects=[Subject(subject=tag) for tag in self.tags],
            descriptions=[
                Description(description=self.description, descriptionType="Other")  # type: ignore
            ]
            if self.description
            else None,
        ).json()
