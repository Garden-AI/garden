from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
from datetime import datetime
from functools import reduce
from inspect import signature
from keyword import iskeyword
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

import dparse  # type: ignore
import globus_compute_sdk  # type: ignore
from packaging.requirements import InvalidRequirement, Requirement
from pydantic import BaseModel, Field, PrivateAttr, root_validator, validator
from pydantic.dataclasses import dataclass

import garden_ai
from garden_ai.app.console import console
from garden_ai.constants import GardenConstants
from garden_ai.datacite import (
    Contributor,
    Creator,
    DataciteSchema,
    Description,
    Identifier,
    Subject,
    Title,
    Types,
)
from garden_ai.mlmodel import ModelMetadata
from garden_ai.steps import DataclassConfig, Step
from garden_ai.utils.misc import (
    JSON,
    garden_json_encoder,
    read_conda_deps,
    safe_compose,
)

logger = logging.getLogger()


@dataclass(config=DataclassConfig)
class Pipeline:
    """The `Pipeline` class represents a sequence of simpler `steps` composed \
    together to perform a more complex task, typically running inference with a \
    pretrained AI/ML Model.

    See also: [RegisteredPipeline][garden_ai.pipelines.RegisteredPipeline]

    Attributes:
        title:
            Human-readable title, as should appear in citations. (required)
        authors:
            The main researchers involved in producing the Pipeline. At least \
            one author is required in order to register a DOI. Personal name \
            format should be: "Family, Given". Order is preserved. (at least one required)
        year:
            Year that should appear in citations. Required, defaults to current year.
        steps:
            Pipeline's steps in the order they should be invoked. Input/output \
            type annotations must agree. (at least one required)
        contributors:
            Acknowledge contributors to the development of this pipeline. These\
            should be distinct from `authors`.
        description:
            Human-readable description of the pipeline.
        short_name:
            Python identifier (i.e. variable name) to be used when accessing the \
            pipeline as an attribute on a Garden, e.g. \
            `my_garden.pipeline_short_name(...)`. Inferred from pipeline source \
            code if not specified.
        version:
            optional, defaults to "0.0.1".
        tags:
            tags, keywords or key phrases pertaining to the pipeline.
        requirements_file:
            full path/to/requirements.txt containing any additional dependencies\
            of the pipeline. Dependencies should be pinned.
        python_version:
            If set, the version of python to use in the container. If unset, \
            tries to infer the version used by models in the pipeline or the \
            user's current version as a fallback.
        doi:
            Should not be set by users. DOI minted with DataCite.
        func_uuid:
            Should not be set by users. Globus Compute function UUID corresponding to \
            the pipeline's composed steps.
        pip_dependencies:
            Optional, populated by contents of `requirements_file` if specified. \
            Contains the currently installed version of the garden-ai package by \
            default, so that the sdk is pinned in the container.
        conda_dependencies:
            Optional, populated by `requirements_file` if it points to a \
            conda.yml environment file.
        model_full_names:
            Optional, collected from steps' metadata.
    """

    title: str = Field(...)
    authors: List[str] = Field(...)
    steps: Tuple[Step, ...] = Field(...)
    contributors: List[str] = Field(default_factory=list, unique_items=True)
    doi: str = Field(...)
    func_uuid: Optional[UUID] = Field(None)
    description: Optional[str] = Field(None)
    version: Optional[str] = "0.0.1"
    year: str = Field(default_factory=lambda: str(datetime.now().year))
    tags: List[str] = Field(default_factory=list, unique_items=True)
    requirements_file: Optional[str] = Field(None)
    python_version: Optional[str] = Field(None)
    pip_dependencies: List[str] = Field(default_factory=list)
    conda_dependencies: List[str] = Field(default_factory=list)
    model_full_names: List[str] = Field(default_factory=list)
    short_name: Optional[str] = Field(None)

    def _composed_steps(*args, **kwargs):
        """ "This method intentionally left blank"

        We define this as a stub here, instead setting it as an attribute in
        `__post_init_post_parse__`, which is the earliest point after we
        validate that the steps are composable that we could modify the Pipeline
        object.
        This indirection is only necessary because `__call__` itself is looked
        up at the class level, so can't be set dynamically for different instances.
        """
        raise NotImplementedError

    @root_validator(pre=True)
    def doi_omitted(cls, values):
        assert "doi" in values, (
            "It seems like no DOI has been minted yet for this `Pipeline`. If you were trying to create a new `Pipeline`, "
            "use `GardenClient.create_pipeline` to initialize a publishable `Pipeline` with a draft DOI."
        )
        return values

    @validator("short_name")
    def is_valid_identifier(cls, name: Optional[str]) -> Optional[str]:
        if name:
            assert name.isidentifier(), "short_name must be a valid python identifier"
            assert not iskeyword(name), "short_name must not be a reserved keyword"
        return name

    @validator("steps")
    def check_steps_composable(cls, steps):
        if len(steps) == 0:
            raise ValueError("Cannot have no steps in a pipeline.")
        try:
            reduce(safe_compose, reversed(steps))
        except TypeError as e:
            logger.error(e)
            raise
        return steps

    @validator("requirements_file")
    def check_exists(cls, req_file):
        if not req_file:
            msg = (
                "No requirements file specified for pipeline. If this "
                "pipeline has any further dependencies, please specify a "
                "requirements file by including a"
                ' `requirements_file="path/to/requirements.txt"` '
                "keyword argument in the constructor."
            )
            logger.warning(msg)
            return None

        req_path = pathlib.Path(req_file).resolve()
        if not req_path.exists():
            raise ValueError(f"Could not find {req_path}. Please make sure it exists.")
        return str(req_path)

    @validator("requirements_file")
    def check_extension(cls, req_file):
        if not req_file:
            # no warning if missing, would be redundant with check_exists
            return None
        if not req_file.endswith((".txt", ".yml", ".yaml")):
            raise ValueError(
                "Did not recognize requirements file extension. Try "
                "again with a standard requirements.txt or conda "
                "environment.yml/yaml."
            )
        return req_file

    @validator("pip_dependencies", each_item=True)
    def pip_deps_parsable(cls, pip_dep):
        try:
            _ = Requirement(pip_dep)
        except InvalidRequirement as e:
            raise ValueError(f"Could not parse pip dependency '{pip_dep}'") from e
        return pip_dep

    @validator("pip_dependencies", each_item=False)
    def ensure_minimal_dependencies(cls, pip_deps):
        import mlflow  # type: ignore

        if not any(req.startswith("mlflow") for req in pip_deps):
            pip_deps += [f"mlflow-skinny=={mlflow.__version__}"]
        if not any(req.startswith("pandas") for req in pip_deps):
            pip_deps += ["pandas<3"]
        return pip_deps

    def _collect_requirements(self):
        """collect requirements to pass to Globus Compute container service.

        Populates attributes `self.python_version, self.pip_dependencies, self.conda_dependencies` per
        `self.requirements_file`, as well as `self.model_full_names` from steps' metadata.
        """
        # collect explicit pipeline dependencies for the container
        if self.requirements_file:
            if self.requirements_file.endswith((".yml", ".yaml")):
                py_version, conda_deps, pip_deps = read_conda_deps(
                    self.requirements_file
                )
                if py_version:
                    self.python_version = py_version
                self.conda_dependencies += conda_deps
                self.pip_dependencies += pip_deps

            elif self.requirements_file.endswith(".txt"):
                with open(self.requirements_file, "r") as f:
                    contents = f.read()
                    parsed = dparse.parse(
                        contents, path=self.requirements_file, resolve=True
                    ).serialize()
                    deps = [d["line"] for d in parsed["dependencies"]]
                    deps.extend(d["line"] for d in parsed["resolved_dependencies"])
                    self.pip_dependencies += set(deps)

        for step in self.steps:
            self.model_full_names += step.model_full_names

        system_py_version = ".".join(map(str, sys.version_info[:3]))
        self.python_version = self.python_version or system_py_version
        self.conda_dependencies = list(set(self.conda_dependencies))
        self.pip_dependencies = list(set(self.pip_dependencies))

        if self.python_version:
            if self.python_version != system_py_version:
                logger.warning(
                    f"Pipeline specified to run with Python version {self.python_version} "
                    f"but Garden is running under {system_py_version}. Using the Python version "
                    f"{self.python_version} specified in the Pipeline."
                )
        return

    def __call__(
        self,
        *args: Any,
        garden_client: garden_ai.GardenClient = None,
        **kwargs: Any,
    ) -> Any:
        """Call the pipeline's composed steps on the given input data.

        To run a Pipeline on a remote endpoint, see ``RegisteredPipeline``.

        Args:
            *args : Any
                Input data passed through the first step in the pipeline
            **kwargs : Any
                Additional keyword arguments passed directly to the first step in the pipeline.
        Returns:
            Results from the pipeline's composed steps called with the given input data.

        Raises:
            Exception:
                Any exception raised over the course of executing the pipeline's composed steps.
        """
        has_models = len(self.model_full_names) > 0
        if has_models:
            if not garden_client:
                raise ValueError("Missing required kwarg 'garden_client'")
            pipeline_url_json = garden_client.generate_presigned_urls_for_pipeline(self)
            os.environ[GardenConstants.URL_ENV_VAR_NAME] = pipeline_url_json

        return self._composed_steps(*args, **kwargs)

    def __post_init_post_parse__(self):
        # Finish initializing the pipeline after validators have run.
        # - Build a single composite function from this pipeline's steps
        # - Update metadata like signature, authors w/r/t underlying steps
        # - Infer conda and pip dependencies from steps and requirements file
        self._composed_steps = reduce(safe_compose, reversed(self.steps))
        self.__signature__ = signature(self._composed_steps)
        self._sync_author_metadata()
        self._collect_requirements()
        return

    def _sync_author_metadata(self):
        known_authors = set(self.authors)
        known_contributors = set(self.contributors)
        for step in self.steps:
            new_contributors = set(step.authors) | set(step.contributors)
            known_contributors |= new_contributors - known_authors
        self.contributors = list(known_contributors)
        return

    def json(self) -> JSON:
        """Helper: serialize pipeline metadata to JSON string."""
        self._sync_author_metadata()
        return json.dumps(self, default=garden_json_encoder)

    def datacite_json(self) -> JSON:
        """Parse this `Pipeline`'s metadata into a DataCite-schema-compliant JSON string."""

        # Leverages a pydantic class `DataCiteSchema`, which was automatically generated from:
        # https://github.com/datacite/schema/blob/master/source/json/kernel-4.3/datacite_4.3_schema.json
        #
        # The JSON returned by this method would be the "attributes" part of a DataCite request body.

        self._sync_author_metadata()
        return DataciteSchema(
            identifiers=[Identifier(identifier=self.doi, identifierType="DOI")],
            types=Types(resourceType="AI/ML Pipeline", resourceTypeGeneral="Software"),  # type: ignore
            creators=[Creator(name=name) for name in self.authors],
            titles=[Title(title=self.title)],
            publisher="thegardens.ai",
            publicationYear=self.year,
            subjects=[Subject(subject=tag) for tag in self.tags],
            contributors=[
                Contributor(name=name, contributorType="Other")  # type: ignore
                for name in self.contributors
            ],
            version=self.version,
            descriptions=[
                Description(description=self.description, descriptionType="Other")  # type: ignore
            ]
            if self.description
            else None,
        ).json()

    def dict(self) -> Dict[str, Any]:
        """Helper: serialize pipeline metadata to dictionary."""
        d = {}
        for key in self.__dataclass_fields__:
            val = getattr(self, key)
            if key == "steps":
                val = [s.dict() for s in val]
            d[key] = val
        return d


class RegisteredPipeline(BaseModel):
    """Metadata of a completed and registered `Pipeline` object. Can be added to a Garden and execute on a remote Globus Compute endpoint.

    Unlike `Pipelines`, `RegisteredPipelines` can be described completely by JSON (since they don't need direct references to functions).

    Note:
        Attributes are nearly identical to [Pipeline][garden_ai.pipelines.Pipeline], with a few exceptions.

    Attributes:
        title:
        authors:
        year:
        steps (list[dict]):
            metadata of pipeline steps, rather than steps themselves.
        contributors:
        description:
        short_name:
        version:
        tags:
        python_version:
        doi:
        func_uuid:
        pip_dependencies:
        conda_dependencies:
        model_full_names:
    """

    doi: str = Field(...)
    func_uuid: Optional[UUID] = Field(...)
    title: str = Field(...)
    short_name: str = Field(...)
    authors: List[str] = Field(...)
    # NOTE: steps are dicts here, not Step objects
    steps: List[Dict[str, Union[str, None, List]]] = Field(...)
    contributors: List[str] = Field(default_factory=list, unique_items=True)
    description: Optional[str] = Field(None)
    version: str = "0.0.1"
    year: str = Field(default_factory=lambda: str(datetime.now().year))
    tags: List[str] = Field(default_factory=list, unique_items=True)
    python_version: Optional[str] = Field(None)
    pip_dependencies: List[str] = Field(default_factory=list)
    conda_dependencies: List[str] = Field(default_factory=list)
    _env_vars: Dict[str, str] = PrivateAttr(default_factory=dict)
    model_full_names: List[str] = Field(default_factory=list)

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

        if self._env_vars:
            # see: utils._meta.inject_env_kwarg
            kwargs = dict(kwargs)
            kwargs["_env_vars"] = self._env_vars

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

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline) -> RegisteredPipeline:
        """Helper: instantiate a `RegisteredPipeline` directly from a `Pipeline` instance.

        Raises:
            ValidationError:
                If any fields required by `RegisteredPipeline` but not \
                `Pipeline` (e.g. `doi`, `func_uuid`) are not set.
        """
        # note: we want every RegisteredPipeline to be re-constructible
        # from mere json, so as a sanity check we use pipeline.json() instead of
        # pipeline.dict() directly
        record = pipeline.json()
        data = json.loads(record)
        return cls(**data)

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
