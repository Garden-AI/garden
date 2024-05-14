from git import Repo  # type: ignore
from git.repo.fun import is_git_dir
from garden_ai.mlmodel import ModelMetadata
from garden_ai.utils.misc import trackcalls
from requests.exceptions import HTTPError
import os
import sys
import requests


class GitHubConnector:
    def __init__(
        self,
        repo_url: str,
        branch="main",
        revision=None,
        local_dir=None,
        enable_imports=True,
    ):
        self.repo_url = repo_url
        self.branch = branch
        self.revision = revision
        self.local_dir = local_dir or "gh_model"
        self.enable_imports = enable_imports
        self.metadata = ModelMetadata(
            model_identifier=self.repo_url,
            model_repository="GitHub",
            model_version=self.revision,
        )

        repo_url_split = repo_url.split("/")

        # Grab the latest commit hash if none was given
        if self.revision is None:
            try:
                # get commit info from GitHub API: https://docs.github.com/en/rest/commits/commits?apiVersion=2022-11-28
                commit_url = f"https://api.github.com/repos/{repo_url_split[-2]}/{repo_url_split[-1]}/commits/{self.branch}"
                response = requests.get(commit_url)
                response.raise_for_status()
                commit_info = response.json()
                commit_hash = commit_info["sha"]
                if commit_hash:
                    self.revision = commit_hash
                    self.metadata.model_version = self.revision
            except Exception as e:
                print(f"Error fetching commit hash: {e}")
                print("Check that the repo_url is correct and that the repo is public.")

        try:
            readme_url = f"https://raw.githubusercontent.com/{repo_url_split[-2]}/{repo_url_split[-1]}/{self.branch}/README.md"
            self.read_me = requests.get(readme_url).text
        except HTTPError:
            self.read_me = ""
        except requests.RequestException:
            self.read_me = ""

    @trackcalls
    def stage(self) -> str:

        if is_git_dir(f"{self.local_dir}/.git"):
            # double check the existing repo in local_dir refers to the same
            # repo as this connector before pulling
            found_repo = Repo(self.local_dir)
            if self.repo_url not in found_repo.remotes.origin.url:
                raise ValueError(
                    f"Failed to clone {self.repo_url} to {self.local_dir} "
                    f"({found_repo.remotes.origin.url} already cloned here)."
                )
            else:
                found_repo.remotes.origin.pull(self.branch)
                return self.local_dir

        Repo.clone_from(f"{self.repo_url}.git", self.local_dir, branch=self.branch)

        # Checkout the pinned revision if available
        if self.revision:
            try:
                repo = Repo(self.local_dir)
                repo.git.checkout(self.revision)
            except GitCommandError:
                raise ValueError(f"Failed to checkout revision {self.revision}")

        if self.enable_imports:
            sys.path.append(self.local_dir)
        return self.local_dir

    def _repr_html_(self):
        if not self.read_me:
            return ""
        try:
            __IPYTHON__  # Check if running in notebook. '__IPYTHON__' is defined if in one.
            from IPython.display import display, Markdown  # type: ignore

            display(Markdown(self.read_me), display_id=True)
            return (
                ""  # we need to return a string do it doesn't go try other repr methods
            )
        except NameError:
            return self.read_me
