from typing import Optional
import huggingface_hub as hfh


class HFConnector:
    def __init__(self, repo_id: str, revision=None, local_dir=None):
        self.repo_id = repo_id
        self.revision = revision
        self.local_dir = local_dir or f"./{repo_id}"
        self.model_card = hfh.ModelCard.load(repo_id)

    def download(self):
        hfh.snapshot_download(
            self.repo_id,
        )

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
