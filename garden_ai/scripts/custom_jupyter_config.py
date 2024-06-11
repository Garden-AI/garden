import pickle
import nbformat
from garden_ai.notebook_metadata import NotebookMetadata
import os.path


# Jupyter post_save_hook to save notebooks metadata whenever the notebook is saved.
def post_save_hook(model, os_path, contents_manager):
    # Check if the saved file is a notebook
    if model["type"] == "notebook":
        with open(os_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # If notebook_metadata.pkl does not exist,
        # no edits to the notebooks metadata have been made with widget, so just exit
        if not os.path.isfile("./notebook_metadata.pkl"):
            return

        # Load picked metadata and save to notebooks metadata
        try:
            with open("./notebook_metadata.pkl", "rb") as f:
                nb_meta = pickle.load(f)

            assert isinstance(nb_meta, NotebookMetadata)

            nb["metadata"]["garden_metadata"] = nb_meta.as_dict()
            nbformat.write(nb, os_path, version=nbformat.NO_CONVERT)
        except FileNotFoundError:
            print("Unable to save notebook metadata")
