import huggingface_hub as hfh  # type: ignore

from .model_connector import ModelConnector
from .exceptions import ConnectorInvalidRevisionError


class HFConnector(ModelConnector):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if "huggingface" not in self.metadata.model_repository:
            raise ValueError("repo_url must be a Hugging Face repository.")

    def _build_url_from_id(self) -> str:
        """Return the full URL to the repo on Hugging Face."""
        return f"https://huggingface.co/{self.repo_id}"

    def _download(self) -> str:
        """Snapshot download the model.

        See: https://huggingface.co/docs/huggingface_hub/en/guides/download#download-an-entire-repository
        """
        if not self.local_dir.exists():  # type: ignore[union-attr]
            self.local_dir.mkdir(parents=True)  # type: ignore[union-attr]

        hfh.snapshot_download(
            repo_id=self.repo_id,
            local_dir=str(self.local_dir),
            revision=self.revision,
        )

        return str(self.local_dir)

    def _fetch_readme(self) -> str:
        """Fetch the repo's model card."""
        try:
            # This fetches README.md from the repo and will raise an error if it doesn't exist
            return hfh.ModelCard.load(self.repo_id).text
        except Exception:
            return ""

    def _infer_revision(self) -> str:
        """Return the latest commit on main."""
        try:
            refs = hfh.list_repo_refs(self.repo_id)
            for branch in refs.branches:
                if branch.name == "main":
                    return branch.target_commit
            return ""
        except Exception as e:
            # just pass along any error message, the list_repo_refs exceptions have helpful messages
            raise ConnectorInvalidRevisionError(e)

    def _checkout_revision(self):
        """Doesn't do anything."""
        # Revision is checked out in _download
        pass
