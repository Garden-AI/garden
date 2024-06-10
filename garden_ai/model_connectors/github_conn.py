from git import Repo, GitCommandError  # type: ignore
import base64
import requests

from .exceptions import (
    ConnectorAPIError,
    ConnectorRevisionError,
    ConnectorInvalidRevisionError,
    ConnectorLFSError,
)
from .model_connector import ModelConnector


class GitHubConnector(ModelConnector):
    """Connect to a model stored on GitHub."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if "github" not in self.metadata.model_repository:
            raise ValueError("repo_url must be a GitHub repository.")

        # Raise an error if git lfs is used by the repo
        if self._has_lfs_file():
            raise ConnectorLFSError()

    def _has_lfs_file(self) -> bool:
        """Check if the repo has any git-lfs files.

        Pulls .gitattributes from the repo and looks for "filter=lfs"
        """

        owner, repo = str(self.repo_id).split("/")

        # git-lfs marks files in .gitattributes with filter=lfs
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/.gitattributes"
        headers = {"Accept": "application/vnd.github+json"}

        try:
            response = requests.get(url, headers)
            if response.status_code == 404:
                # .gitattributes was not found, repo is not using git-lfs
                return False
            response.raise_for_status()
            # file contents are in the 'content' field encoded as base64
            file_data = response.json()["content"]
            contents = base64.b64decode(file_data).decode("utf-8")
            return any("filter=lfs" in line.strip() for line in contents.split("\n"))
        except requests.RequestException as e:
            raise ConnectorAPIError(e)

    def _download(self) -> str:
        """Clone the repo into self.local_dir"""
        Repo.clone_from(f"{self.repo_url}.git", str(self.local_dir), branch=self.branch)
        return str(self.local_dir)

    def _fetch_readme(self) -> str:
        """Attempt to retrieve README.md from remote repo.

        Returns: str README.md text or ''
        """
        owner, repo = str(self.repo_id).split("/")
        readme_url = (
            f"https://raw.githubusercontent.com/{owner}/{repo}/{self.branch}/README.md"
        )
        response = requests.get(f"{readme_url}")
        if response.status_code == 200:
            return response.text
        else:
            return ""

    def _infer_revision(self) -> str:
        """Get the commit hash from the HEAD of main.

        Raises:
            ConnectorInvalidRevisionError: when a commit hash cannot be found.
        """
        owner, repo = str(self.repo_id).split("/")
        try:
            # get commit info from GitHub API: https://docs.github.com/en/rest/commits/commits?apiVersion=2022-11-28
            commit_url = (
                f"https://api.github.com/repos/{owner}/{repo}/commits/{self.branch}"
            )
            response = requests.get(commit_url)
            response.raise_for_status()
            commit_info = response.json()
            commit_hash = commit_info["sha"]
            if commit_hash:
                return commit_hash
            else:
                return ""
        except Exception as e:
            raise ConnectorInvalidRevisionError(
                e, "Is the repo_url correct and the repo public?"
            )

    def _checkout_revision(self):
        """Checkout the pinned revision if available.

        Raises:
            ValueError: when there is an error checking out the revision.
        """
        try:
            repo = Repo(self.local_dir)
            repo.git.checkout(self.revision)
        except GitCommandError as e:
            raise ConnectorRevisionError(
                f"Failed to checkout revision {self.revision}"
            ) from e

    def _build_url_from_id(self) -> str:
        """Return the full GitHub url to the repo."""
        return f"https://github.com/{self.repo_id}"
