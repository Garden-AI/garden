import huggingface_hub as hfh  # type: ignore
from garden_ai.mlmodel import ModelMetadata
from garden_ai.utils.misc import trackcalls
from requests.exceptions import HTTPError
import os
import sys

from .exceptions import ConnectorInvalidRevisionError


class HFConnector:
    def __init__(
        self, repo_id: str, revision=None, local_dir=None, enable_imports=True
    ):
        self.repo_id = repo_id
        self.revision = revision
        self.local_dir = local_dir or "hf_model"
        self.enable_imports = enable_imports
        self.metadata = ModelMetadata(
            model_identifier=self.repo_id,
            model_repository="Hugging Face",
            model_version=self.revision,
        )
        try:
            # This fetches README.md from the repo and will raise an error if it doesn't exist
            self.model_card = hfh.ModelCard.load(repo_id)
        except HTTPError:
            self.model_card = None

        # Grab the model revision from the main branch if it wasn't provided.
        if self.revision is None:
            try:
                refs = hfh.list_repo_refs(self.repo_id)
                for branch in refs.branches:
                    if branch.name == "main":
                        self.revision = branch.target_commit
                        self.metadata.model_version = self.revision
                assert self.revision is not None
            except Exception as e:
                # just pass along any error message, the list_repo_refs exceptions have helpful messages
                raise ConnectorInvalidRevisionError(e)

    @trackcalls
    def stage(self) -> str:
        if not os.path.exists(self.local_dir):
            os.mkdir(self.local_dir)
        hfh.snapshot_download(
            self.repo_id, revision=self.revision, local_dir=self.local_dir
        )
        if self.enable_imports:
            sys.path.append(self.local_dir)
        return self.local_dir

    def _repr_html_(self):
        if not self.model_card:
            return ""
        try:
            __IPYTHON__  # Check if running in notebook. '__IPYTHON__' is defined if in one.
            from IPython.display import display, Markdown  # type: ignore

            display(Markdown(self.model_card.text), display_id=True)
            return (
                ""  # we need to return a string do it doesn't go try other repr methods
            )
        except NameError:
            return self.model_card.text
