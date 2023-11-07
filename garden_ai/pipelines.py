from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, List, Optional, Union
from uuid import UUID

import globus_compute_sdk  # type: ignore
from pydantic import BaseModel, Field

# from garden_ai.app.console import console
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
    """Metadata for a pipeline prior to its registration. \
    Passed to the `garden_pipeline` decorator during the registration process.

    Use the optional doi field if you have already registered this pipeline \
    and want to update it under the same DOI.

    Attributes:
        doi: Optional. If you have a DOI you want to use for this pipeline, specify it here. \
        (Especially if you have already registered a pipeline and are updating it.) \
        Otherwise we will generate a DOI for you.
        title: A short title that describes the pipeline.
        description: A longer free text description of this pipeline.
        authors: A list of the authors of this pipeline. You need at least one.
        short_name: This will be the name of the Python method that people call to invoke your pipeline.
        year: When did you make this pipeline? (Defaults to current year)
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


class RegisteredPipeline(PipelineMetadata):
    """Metadata of a completed and registered pipeline.
    Can be added to a Garden and executed on a remote Globus Compute endpoint.

    Has all the user-given metadata from PipelineMetadata plus extra fields added by Garden
    during publication.

    Attributes:
        func_uuid: The ID of the Globus Compute function registered for this pipeline.
        container_uuid: ID returned from Globus Compute's register_container.
        base_image_name: The name of the base image selected by the user. eg, "3.9-base"
        base_image_uri: Name and location of the base image used by this pipeline. eg docker://index.docker.io/maxtuecke/garden-ai:python-3.9-jupyter
        full_image_uri: The name and location of the complete image used by this pipeline.
        notebook: Full JSON string of the notebook used to define this pipeline's environment.
    """

    doi: str = Field(
        ...
    )  # Repeating this field from base class because DOI is mandatory for RegisteredPipeline
    func_uuid: UUID = Field(...)
    container_uuid: UUID = Field(...)
    base_image_name: Optional[str] = Field(None)
    base_image_uri: Optional[str] = Field(None)
    full_image_uri: Optional[str] = Field(None)
    notebook: Optional[str] = Field(None)

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
            future = gce.submit_to_registered_function(
                function_id=str(self.func_uuid), args=args, kwargs=kwargs
            )
            return future.result()

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


def garden_pipeline(
    metadata: PipelineMetadata,
    garden_doi: str = None,
    model_connectors=None,
):
    def decorate(func):
        # let func carry its own metadata
        func._pipeline_meta = metadata.dict()
        func._model_connectors = model_connectors or []
        func._garden_doi = garden_doi
        return func

    return decorate
