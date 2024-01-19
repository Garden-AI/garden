from git import Repo  # type: ignore
from garden_ai.mlmodel import ModelMetadata
from garden_ai.utils.misc import trackcalls
from requests.exceptions import HTTPError
import os
import sys
import requests


class GitHubConnector:
    def __init__(
        self, repo_url: str, revision=None, local_dir=None, enable_imports=True
    ):
        self.repo_url = repo_url
        self.revision = revision
        self.local_dir = local_dir or "gh_model"
        self.enable_imports = enable_imports
        self.metadata = ModelMetadata(
            model_identifier=self.repo_url,
            model_repository="GitHub",
            model_version=self.revision,
        )
        try:
            repo_url_split = repo_url.split("/")
            readme_url = f"https://raw.githubusercontent.com/{repo_url_split[-2]}/{repo_url_split[-1]}/main/README.md"
            self.read_me = requests.get(readme_url)
        except HTTPError:
            self.read_me = None
        except requests.RequestException:
            self.read_me = None

    @trackcalls
    def stage(self) -> str:  # TODO Review this
        if not os.path.exists(self.local_dir):
            os.mkdir(self.local_dir)

        Repo.clone_from(f"{self.repo_url}.git", self.local_dir)

        if self.enable_imports:
            sys.path.append(self.local_dir)
        return self.local_dir

    def _repr_html_(self):
        if not self.read_me:
            return ""
        try:
            __IPYTHON__  # Check if running in notebook. '__IPYTHON__' is defined if in one.
            from IPython.display import display, Markdown  # type: ignore

            display(Markdown(self.read_me.text), display_id=True)
            return (
                ""  # we need to return a string do it doesn't go try other repr methods
            )
        except NameError:
            return self.read_me.text
