from __future__ import annotations

import json
import logging
import pathlib
import sys
from datetime import datetime
from functools import reduce
from inspect import signature
from typing import Any, List, Optional, Tuple, cast
from uuid import UUID, uuid4

import dparse  # type: ignore
from pydantic import Field, validator
from pydantic.dataclasses import dataclass

from garden_ai.datacite import (
    Contributor,
    Creator,
    DataciteSchema,
    Description,
    Title,
    Types,
)
from garden_ai.steps import DataclassConfig, Step
from garden_ai.utils import garden_json_encoder, read_conda_deps, safe_compose

logger = logging.getLogger()


@dataclass(config=DataclassConfig)
class Pipeline:
    """
    The ``Pipeline`` class represents a sequence of steps
    that form a pipeline. It has a list of authors, a title,
    and a list of steps. The __call__ method can be used
    to execute the pipeline by calling each Step in order
    with the output of the previous Step as the input to the
    next Step. The register method can be used to register
    each Step in the pipeline.

    Args:
    authors (List[str]): A list of the authors of the pipeline.
    title (str): The title of the pipeline.
    steps (List[Step]): A list of the steps in the pipeline.

    """

    title: str = Field(...)
    authors: List[str] = Field(...)
    steps: Tuple[Step, ...] = Field(...)
    contributors: List[str] = Field(default_factory=list, unique_items=True)
    doi: str = cast(str, Field(default_factory=lambda: None))
    uuid: UUID = Field(default_factory=uuid4)
    description: Optional[str] = Field(None)
    version: str = "0.0.1"
    year: str = Field(default_factory=lambda: str(datetime.now().year))
    tags: List[str] = Field(default_factory=list, unique_items=True)
    requirements_file: str = "./requirements.txt"
    python_version: Optional[str] = Field(None)
    pip_dependencies: List[str] = Field(default_factory=list)
    conda_dependencies: List[str] = Field(default_factory=list)

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
        req_file = pathlib.Path(req_file)
        if not req_file.exists():
            raise ValueError(f"Could not find {req_file}. Please make sure it exists.")
        return str(req_file)

    @validator("requirements_file")
    def check_extension(cls, req_file):
        if not req_file.endswith((".txt", ".yml", ".yaml")):
            raise ValueError(
                "Did not recognize requirements file extension. Try"
                "again with a standard requirements.txt or conda"
                "environment.yml/yaml."
            )
        return req_file

    def _collect_requirements(self):
        """collect requirements to pass to funcx container service.

        Populates attributes: ``self.python_version, self.pip_dependencies, self.conda_dependencies``
        """

        # mapping of python-version-witness: python-version (for warning msg)
        py_versions = {
            "system": ".".join(map(str, sys.version_info[:3])),
            "pipeline": self.python_version,
        }
        if self.requirements_file.endswith((".yml", ".yaml")):
            py_version, conda_deps, pip_deps = read_conda_deps(self.requirements_file)
            if py_version:
                py_versions["pipeline"] = py_version
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
                self.pip_dependencies += deps

        for step in self.steps:
            self.conda_dependencies += step.conda_dependencies
            self.pip_dependencies += step.pip_dependencies
            py_versions[step.__name__] = step.python_version

        self.python_version = py_versions["pipeline"] or py_versions["system"]
        self.conda_dependencies = list(set(self.conda_dependencies))
        self.pip_dependencies = list(set(self.pip_dependencies))

        if len(set(py_versions[k] for k in py_versions if py_versions[k])) > 1:
            logger.warning(
                "Found multiple python versions specified across this"
                f"pipeline's dependencies: {py_versions}. {self.python_version} "
                "will be used by default. This version can be set explicitly via "
                "the `python_version` keyword argument in the `Pipeline` "
                "constructor."
            )
        return

    def register(self):
        """register this Pipeline's complete Step composition as a funcx function
        TODO
        """
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._composed_steps(*args, **kwargs)

    def __post_init_post_parse__(self):
        """Finish initializing the pipeline after validators have run.

        - Build a single composite function from this pipeline's steps
        - Update metadata like signature, authors w/r/t underlying steps
        - Infer conda and pip dependencies from steps and requirements file
        """
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

    def json(self):
        return json.dumps(self, default=garden_json_encoder)

    def datacite_json(self):
        """Parse this `Pipeline`'s metadata into a DataCite-schema-compliant JSON string.

        Leverages a pydantic class `DataCiteSchema`, which was automatically generated from:
        https://github.com/datacite/schema/blob/master/source/json/kernel-4.3/datacite_4.3_schema.json

        The JSON returned by this method would be the "attributes" part of a DataCite request body.
        """
        self._sync_author_metadata()
        return DataciteSchema(
            types=Types(resourceType="AI/ML Pipeline", resourceTypeGeneral="Software"),  # type: ignore
            creators=[Creator(name=name) for name in self.authors],
            titles=[Title(title=self.title)],
            publisher="thegardens.ai",
            publicationYear=self.year,
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
