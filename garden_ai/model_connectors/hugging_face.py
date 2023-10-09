import huggingface_hub as hfh
from garden_ai.mlmodel import ModelMetadata
import os


class HFConnector:
    def __init__(self, repo_id: str, revision=None, local_dir=None, flavor="sklearn"):
        self.repo_id = repo_id
        self.revision = revision
        self.local_dir = local_dir or "hf_model"
        self.model_card = hfh.ModelCard.load(repo_id)
        user_part, model_part = self.repo_id.split("/")
        self.metadata = ModelMetadata(
            model_name=model_part, user_email=user_part, flavor=flavor
        )

    def download(self):
        if not os.path.exists(self.local_dir):
            os.mkdir(self.local_dir)
        hfh.snapshot_download(self.repo_id, local_dir=self.local_dir)

    def _repr_html_(self):
        try:
            __IPYTHON__  # Check if running in notebook. '__IPYTHON__' is defined if in one.
            from IPython.display import display, Markdown

            display(Markdown(self.model_card.text), display_id=True)
            return (
                ""  # we need to return a string do it doesn't go try other repr methods
            )
        except NameError:
            return self.model_card.text
