from abc import ABC, abstractmethod
from pathlib import Path
import re
import sys
from typing import Optional

from git import Repo
from git.repo.fun import is_git_dir

from pydantic import (
    BaseModel,
    HttpUrl,
    root_validator,
    validator,
)

from garden_ai.mlmodel import ModelMetadata, match_repo_type

from garden_ai.utils.misc import trackcalls

from .exceptions import (
    ConnectorStagingError,
    ConnectorInvalidRevisionError,
)


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
        model_dir: base directory for model downloads. defaults to './moodels'
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

        # Raises an error if unsupported repo type
        repo_type = match_repo_type(self.repo_url)

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
        """Fetch the README or equivalent from the remote repo."""
        raise NotImplementedError()

    @abstractmethod
    def _build_url_from_id(repo_id: str) -> str:
        """Return a full URL to the repo from the repo_id."""
        raise NotImplementedError()

    @abstractmethod
    def _checkout_revision(self):
        """Checkout self.revision."""
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

    def _repr_html_(self):
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
    def check_repo_identifier(cls, values):
        """Make sure either repo_url or repo_id are passed to the constructor.

        At least one is needed.
        """
        repo_url = values.get("repo_url")
        repo_id = values.get("repo_id")

        if repo_url is None and repo_id is None:
            raise ValueError("Either repo_url or repo_id must be provided")

        return values

    @root_validator
    def _build_repo_url(cls, values) -> str:
        """Build the full URL to the repo based on the repo_id."""
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
        """Build a repo_id in the form 'owner/repo' from the repo_url."""
        repo_url = values.get("repo_url")
        if repo_url is not None:
            repo_url_split = repo_url.split("/")
            owner, repo = repo_url_split[-2], repo_url_split[-1]
            return f"{owner}/{repo}"
        return v
