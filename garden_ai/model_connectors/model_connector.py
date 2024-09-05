import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from git import Repo
from git.repo.fun import is_git_dir
from pydantic import (
    BaseModel,
    ConfigDict,
    HttpUrl,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from garden_ai.schemas.entrypoint import ModelMetadata
from garden_ai.utils.misc import trackcalls

from .exceptions import (
    ConnectorInvalidRepoIdError,
    ConnectorInvalidRevisionError,
    ConnectorInvalidUrlError,
    ConnectorStagingError,
)


class ModelConnector(BaseModel, ABC):
    """Provides attributes and functionality common to models stored in remote git repos.

    **Model Connectors are not meant to be constructed directly by users.** See: the [create_connector][garden_ai.create_connector] helper function.

    Intended to be subclassed. See `GitHubConnector` and `HFConnector` for example implementations.

    Attributes:
        repo_url: A str URL to the remote repo. e.g. https://huggingface.co/Garden-AI/sklearn-iris
        repo_id: A str identifier for the repo in the form 'owner/repo'. e.g. Garden-AI/sklearn-iris
        branch: The git branch to use. Defaults to 'main'.
        revision: git commit hash to checkout. Defaults to the HEAD of 'main'.
        local_dir: the local directory to download to repo. Defaults to './models/<repo_name>'
        enable_imports: enable Python package imports from local_dir.
        metadata: A `ModelMetadata` object. Will be computed from given information if not provided.
        readme: an optional readme for the repo. Typically fetched automatically from connected repository.
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
    def _parse_id_from_url(url: str) -> Optional[str]:
        """Return a repo id in the form 'owner/repo" from the provided URL."""
        url_parts = str(url).split("/")
        owner, repo = url_parts[-2], url_parts[-1]
        return ModelConnector._validate_repo_id(f"{owner}/{repo}")

    @staticmethod
    def _validate_repo_id(repo_id: str) -> Optional[str]:
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
    def _is_valid_url(repo: Union[HttpUrl, str]) -> Optional[Union[HttpUrl, str]]:
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
        """Download the repository contents to local_dir.

        Should only be called within a `@entrypoint` function, not at the top-level of a notebook.

        Returns:
            Path to the local directory where the model was downloaded.

        Raises:
            ConnectorStagingError: If something goes wrong during staging.
        """
        try:
            # pull from the repo if we have already downloaded it.
            if self._pull_if_downloaded():
                model_dir = str(self.local_dir)
            else:
                # otherwise download the repo
                model_dir = self._download()

            # ensure the correct commit is checked out
            self._checkout_revision()
            return model_dir
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
            from IPython.display import Markdown, display  # type: ignore

            display(Markdown(self.readme), display_id=True)
            return (
                ""  # we need to return a string do it doesn't go try other repr methods
            )
        except NameError:
            return self.readme

    @model_validator(mode="after")
    def _validate_required_fields(self):
        if self.repo_url is None and self.repo_id is None:
            raise ValueError("Must provide either repo_url or repo_id")

        if self.repo_url and self.repo_id is None:
            # we need a repo_id
            self.repo_id = ModelConnector._parse_id_from_url(self.repo_url)
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
    def _validate_revision(cls, revision) -> str:
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
