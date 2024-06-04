from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
import re
import sys
from typing import List, Optional

from git import Repo
from git.repo.fun import is_git_dir

from pydantic import (
    BaseModel,
    HttpUrl,
    root_validator,
    validator,
    Field,
)

from garden_ai.utils.misc import trackcalls

from .exceptions import (
    ConnectorStagingError,
    ConnectorInvalidRevisionError,
)


class ModelRepository(Enum):
    HUGGING_FACE = "Hugging Face"
    GITHUB = "GitHub"

    def match_by_url(url: str):
        if "github.com" in url:
            return ModelRepository.GITHUB
        elif "huggingface.co" in url:
            return ModelRepository.HUGGING_FACE
        else:
            return None


class DatasetConnection(BaseModel):
    """
    The ``DatasetConnection`` class represents metadata of external datasets \
    which publishers want to highlight as related to their Entrypoint. This \
    metadata (if provided) will be displayed with the Entrypoint as "related \
    work".

    These can be linked to an Entrypoint directly in the `EntrypointMetadata` or \
    via the ``@garden_entrypoint`` decorator with the `datasets` kwarg.

    Example:

        ```python
        my_relevant_dataset = DatasetConnection(
            title="Benchmark Dataset for Locating Atoms in STEM images",
            doi="10.18126/e73h-3w6n",
            url="https://foundry-ml.org/#/datasets/10.18126%2Fe73h-3w6n",
            repository="foundry",
        )
        my_metadata = EntrypointMetadata(
            title="...",
            # etc
        )

        @garden_entrypoint(metadata=my_metadata, datasets=[my_relevant_dataset])
        def my_entrypoint(*args, **kwargs):
            ...

        ```


    Attributes:
        title (str):
            A short and descriptive name of the dataset.
        doi (str):
            A digital identifier to the dataset.
        url (str):
            Location where the dataset can be accessed. If using foundry \
            dataset, both url and DOI must be provided.
        repository (str):
            The public repository where the dataset is hosted (e.g. "foundry", "github")
        data_type (str):
            Optional, the type of file of dataset.

    """

    title: str = Field(...)
    doi: Optional[str] = Field(None)
    url: str = Field(...)
    data_type: Optional[str] = Field(None)
    repository: str = Field(...)

    @validator("repository")
    def check_foundry(cls, v, values, **kwargs):
        v = v.lower()  # case-insensitive
        if "url" in values and "doi" in values:
            if v == "foundry" and (values["url"] is None or values["doi"] is None):
                raise ValueError(
                    "For a Foundry repository, both url and doi must be provided"
                )
        return v


class ModelMetadata(BaseModel):
    """
    The ``ModelMetadata`` class represents metadata about an ML model published  \
    on a public model repository used in an Entrypoint.

    Attributes:
        model_identifier (str): A short and descriptive name of the model
        model_repository (ModelRepository): The repository the model is published on.
        model_version (str): A version identifier
        datasets (DatasetConnection):
            One or more dataset records that the model was trained on.
    """

    model_identifier: str = Field(...)
    model_repository: str = Field(...)
    model_version: Optional[str] = Field(None)
    datasets: List[DatasetConnection] = Field(default_factory=list)

    @validator("model_repository")
    def must_be_a_supported_repository(cls, model_repository):
        if model_repository not in [mr.value for mr in ModelRepository]:
            raise ValueError("is not a supported flavor")
        return model_repository


class ModelConnector(BaseModel, ABC):
    """Provides attributes and functionality common to models stored in remote git repos.

    Intended to be sub-classed. See `GitHubConnector` and `HFConnector`
    for example implementations.

    Attributes:
        repo_url: A str URL to the remote repo. e.g. https://huggingface.co/Garden-AI/sklearn-iris
        repo_id: A str identifier for the repo in the form 'owner/repo'. e.g. Garden-AI/sklearn-iris
        branch: str optional git branch to use. Defaults to 'main'.
        revision: str optional git commit hash to checkout. Defaults to the HEAD of 'main'.
        local_dir: the local directory to download to repo. Defaults to './models/<repo_name>'
        enable_imports: enable Python package imports from local_dir.
        metadata: A `ModelMetadata` object. Will be computed from given information if not provided.
        readme: an optional readme for the repo.
        model_dir: base directory for model downloads. defaults to './models'
    """

    repo_url: Optional[HttpUrl]
    repo_id: Optional[str]
    branch: str = "main"
    revision: Optional[str]
    local_dir: Optional[Path]
    enable_imports: bool = True
    metadata: Optional[ModelMetadata]
    readme: Optional[str]
    model_dir: Optional[Path] = "models"

    class Config:
        validate_assignment = True  # Validate assignment to fields after creation

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        repo_type = ModelRepository.match_by_url(self.repo_url)
        if self.metadata is None:
            self.metadata = ModelMetadata(
                model_identifier=str(self.repo_id),
                model_repository=repo_type.value,
                model_version=str(self.revision) or None,
            )

        if self.local_dir is None:
            # default to "./models/repo_name"
            path = Path(f"{self.model_dir}/{self.repo_id.split('/')[-1]}")
            self.local_dir = path

        if self.readme is None:
            self.readme = self._fetch_readme()

        if self.revision is None:
            self.revision = self._infer_revision()
            self.metadata.model_version = str(self.revision)

        if self.enable_imports:
            sys.path.append(self.local_dir)

    @abstractmethod
    def _infer_revision(self) -> str:
        """Infer a revision to tag from the repo.

        Returns:
            A git commit hash str. Git commit hashes are 40-character hex strings.
        """
        raise NotImplementedError()

    @abstractmethod
    def _download(self) -> str:
        """Download the remote repo into self.local_dir.

        Returns:
            A str path to the local directory the repo was downloaded.
        """
        raise NotImplementedError()

    @abstractmethod
    def _fetch_readme(self) -> str:
        """Fetch the README or equivalent from the remote repo.

        Returns:
            A str with the contents of README or equivalent.
        """
        raise NotImplementedError()

    @abstractmethod
    def _build_url_from_id(repo_id: str) -> str:
        """Return a full URL to the repo from the repo_id."""
        raise NotImplementedError()

    @abstractmethod
    def _checkout_revision(self):
        """Checkout self.revision if appropriate."""
        raise NotImplementedError()

    @trackcalls
    def stage(self) -> str:
        """Download the repository from repo_url to local_dir.

        Should only be called within a `@garden_entrypoint` function if running
        in a notebook.

        Returns: a str Path to local directory where the model was downloaded.

        Raises:
            ConnectorStagingError: when something goes wrong during staging.
        """
        try:
            self._checkout_revision()

            # pull from the repo if we have already downloaded it.
            if self._pull_if_downloaded():
                return self.local_dir
            else:
                # otherwise download the repo
                return self._download()

        except Exception as e:
            raise ConnectorStagingError() from e

    def _pull_if_downloaded(self) -> bool:
        """Check if the repo is already present, if so pull from the remote.

        Will always return false if self.local_dir is not a git repo.

        Returns: True if a pull was executed, false otherwise

        Raises:
            ValueError if a different repo is already present.
        """
        if is_git_dir(f"{self.local_dir}/.git"):
            # double check the existing repo in local_dir refers to the same
            # repo as this connector before pulling
            found_repo = Repo(str(self.local_dir))
            if self.repo_url not in found_repo.remotes.origin.url:
                raise ValueError(
                    f"Failed to clone {self.repo_url} to {self.local_dir} "
                    f"({found_repo.remotes.origin.url} already cloned here)."
                )
            else:
                found_repo.remotes.origin.pull(self.branch)
                return True
        return False

    def _repr_html_(self) -> str:
        """Display README as HTML for jupyter notebooks.

        If not running in a notebook, return readme as a str.
        """
        if not self.readme:
            return ""
        try:
            __IPYTHON__  # Check if running in notebook. '__IPYTHON__' is defined if in one.
            from IPython.display import display, Markdown  # type: ignore

            display(Markdown(self.readme), display_id=True)
            return (
                ""  # we need to return a string do it doesn't go try other repr methods
            )
        except NameError:
            return self.readme

    @root_validator()
    def _check_repo_identifier(cls, values) -> dict:
        """Make sure either repo_url or repo_id are passed to the constructor.

        At least one is needed.

        Raises:
            ValueError: When neither repo_url or repo_id is present.
        """
        repo_url = values.get("repo_url")
        repo_id = values.get("repo_id")

        if repo_url is None and repo_id is None:
            raise ValueError("Either repo_url or repo_id must be provided")

        return values

    @root_validator
    def _build_repo_url(cls, values) -> dict:
        """Return a full URL to the repo based on the repo_id."""
        repo_id = values.get("repo_id")
        repo_url = values.get("repo_url")
        if repo_url is None and repo_id is not None:
            values["repo_url"] = cls._build_url_from_id(repo_id)
        return values

    @validator("revision")
    def validate_revision(cls, revision) -> str:
        """Validate that revision is a valid git commit hash.

        Commit hashes are a 40-character hex string.

        Raises:
            ConnectorInvalidRevisionError: when the revision is not a valid commit hash
        """
        if revision is not None:
            if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                raise ConnectorInvalidRevisionError(
                    "revision must be a valid git commit hash"
                )
        return revision

    @validator("repo_id", always=True)
    def _compute_repo_id(cls, v, values) -> str:
        """Return a repo_id in the form 'owner/repo' from the repo_url."""
        repo_url = values.get("repo_url")
        if repo_url is not None:
            repo_url_split = repo_url.split("/")
            owner, repo = repo_url_split[-2], repo_url_split[-1]
            return f"{owner}/{repo}"
        return v
