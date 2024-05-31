import huggingface_hub as hfh  # type: ignore

from .model_connector import ModelConnector
from .exceptions import ConnectorInvalidRevisionError


class HFConnector(ModelConnector):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if "Hugging Face" not in self.metadata.model_repository:
            raise ValueError("repo_url must be a Hugging Face repository.")

    def _build_url_from_id(repo_id: str) -> str:
        return f"https://huggingface.co/{repo_id}"

    def _download(self) -> str:
        if not self.local_dir.exists():
            self.local_dir.mkdir(parents=True)

        hfh.snapshot_download(
            repo_id=self.repo_id,
            local_dir=str(self.local_dir),
            revision=self.revision,
        )

        return str(self.local_dir)

    def _fetch_readme(self) -> str:
        # TODO this needs better exception handling
        try:
            # This fetches README.md from the repo and will raise an error if it doesn't exist
            return hfh.ModelCard.load(self.repo_id).text
        except Exception:
            return ""

    def _infer_revision(self) -> str:
        try:
            refs = hfh.list_repo_refs(self.repo_id)
            for branch in refs.branches:
                if branch.name == "main":
                    return branch.target_commit
        except Exception as e:
            # just pass along any error message, the list_repo_refs exceptions have helpful messages
            raise ConnectorInvalidRevisionError(e)

    def _checkout_revision(self):
        # Revision is checked out in _download
        pass
