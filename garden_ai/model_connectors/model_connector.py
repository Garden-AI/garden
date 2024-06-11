from abc import ABC, abstractmethod
from pathlib import Path
import re
import sys
from typing import List, Optional, Union

from git import Repo
from git.repo.fun import is_git_dir

from pydantic import (
    BaseModel,
    HttpUrl,
    field_validator,
    Field,
    model_validator,
    ValidationError,
    TypeAdapter,
    ConfigDict,
)

from garden_ai.utils.misc import trackcalls

from .exceptions import (
    ConnectorStagingError,
    ConnectorInvalidRevisionError,
    ConnectorInvalidUrlError,
    ConnectorInvalidRepoIdError,
)


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

    @field_validator("repository")
    @classmethod
    def check_foundry(cls, v, values):
        v = v.lower()  # case-insensitive
        if "url" in values.data and "doi" in values.data:
            if v == "foundry" and (
                values.data["url"] is None or values.data["doi"] is None
            ):
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
        model_repository (str): The repository the model is published on.
        model_version (str): A version identifier
        datasets (DatasetConnection):
            One or more dataset records that the model was trained on.
    """

    model_config = ConfigDict(protected_namespaces=())

    model_identifier: str = Field(...)
    model_repository: str = Field(...)
    model_version: Optional[str] = Field(None)
    datasets: List[DatasetConnection] = Field(default_factory=list)


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

    repo_url: Optional[HttpUrl] = None
    repo_id: Optional[str] = None
    branch: str = "main"
    revision: Optional[str] = None
    local_dir: Optional[Path] = None
    enable_imports: bool = True
    metadata: Optional[ModelMetadata] = None
    readme: Optional[str] = None
    model_dir: Optional[Union[Path, str]] = "models"

    model_config = ConfigDict(
        validate_assignment=True,
        protected_namespaces=(),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
    def _build_url_from_id(self) -> str:
        """Return a full URL to the repo from self.repo_id."""
        raise NotImplementedError()

    @abstractmethod
    def _checkout_revision(self):
        """Checkout self.revision if appropriate."""
        raise NotImplementedError()

    @staticmethod
    def parse_id_from_url(url: str) -> Optional[str]:
        """Return a repo id in the form 'owner/repo" from the provided URL."""
        url_parts = str(url).split("/")
        owner, repo = url_parts[-2], url_parts[-1]
        return ModelConnector.validate_repo_id(f"{owner}/{repo}")

    @staticmethod
    def validate_repo_id(repo_id: str) -> Optional[str]:
        """Parse repo_id to make sure it is in the form 'owner/repo'

        Return: repo_id as a string

        Raises:
            ValidationError: When repo_id is not in the form 'owner/repo'
        """
        if re.fullmatch(r"^[a-zA-Z0-9-]+\/[a-zA-Z0-9-]+$", str(repo_id).strip()):
            return repo_id
        else:
            raise ConnectorInvalidRepoIdError(
                f"Invalid repo_id: {repo_id}. Must be in the form 'owner/repo'"
            )

    @staticmethod
    def is_valid_url(repo: Union[HttpUrl, str]) -> Optional[Union[HttpUrl, str]]:
        """Validate the given url.

        Returns: the URL if valid, otherwise None

        Raises:
           ConnectorInvalidUrlError: When the url is not valid.
        """
        try:
            url_adapter = TypeAdapter(HttpUrl)
            return url_adapter.validate_python(repo)
        except ValidationError:
            raise ConnectorInvalidUrlError(
                f"Invalid repo url: {repo}. Repo url must be a valid HTTP URL."
            )

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
                return str(self.local_dir)
            else:
                # otherwise download the repo
                return self._download()

        except Exception as e:
            raise ConnectorStagingError(str(self.repo_url), None) from e

    def _pull_if_downloaded(self) -> bool:
        """Check if the repo is already present in local_dir, if so pull from the remote.

        Will always return false if self.local_dir is not a git repo.

        Returns: True if a pull was executed, false otherwise

        Raises:
            ValueError if a different repo is already present.
        """
        if is_git_dir(f"{self.local_dir}/.git"):
            # double check the existing repo in local_dir refers to the same
            # repo as this connector before pulling
            found_repo = Repo(str(self.local_dir))
            if str(self.repo_url) not in found_repo.remotes.origin.url:
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
            # Check if running in notebook. '__IPYTHON__' is defined if in one.
            __IPYTHON__  # type: ignore[name-defined]
            from IPython.display import display, Markdown  # type: ignore

            display(Markdown(self.readme), display_id=True)
            return (
                ""  # we need to return a string do it doesn't go try other repr methods
            )
        except NameError:
            return self.readme

    @model_validator(mode="after")
    def validate_required_fields(self):
        if self.repo_url is None and self.repo_id is None:
            raise ValueError("Must provide either repo_url or repo_id")

        if self.repo_url and self.repo_id is None:
            # we need a repo_id
            self.repo_id = ModelConnector.parse_id_from_url(self.repo_url)
        elif self.repo_id and self.repo_url is None:
            # we need a url
            self.repo_url = self._build_url_from_id()

        repo_type = self.repo_url.host
        if self.metadata is None:
            self.metadata = ModelMetadata(
                model_identifier=str(self.repo_id),
                model_repository=repo_type,
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

    @field_validator("revision")
    def validate_revision(cls, revision) -> str:
        """Validate that revision is a valid git commit hash.

        Commit hashes are a 40-character hex string.

        Raises:
            ConnectorInvalidRevisionError: when the revision is not a valid commit hash
        """
        if revision is not None:
            if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                raise ConnectorInvalidRevisionError(
                    "revision must be a valid 40-character git commit hash"
                )
        return revision
