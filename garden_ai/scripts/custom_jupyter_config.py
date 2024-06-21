import json
import nbformat
import os.path

from garden_ai.notebook_metadata import NotebookMetadata


# Jupyter post_save_hook to save notebooks metadata whenever the notebook is saved.
def post_save_hook(model, os_path, contents_manager):
    # Check if the saved file is a notebook
    if model["type"] == "notebook":
        # If notebook_metadata.json does not exist,
        # no edits to the notebooks metadata have been made with widget, so just exit
        if not os.path.isfile("./notebook_metadata.json"):
            print("Could not file notebook_metadata.json")
            return

        # Load picked metadata and save to notebooks metadata
        with open("./notebook_metadata.json", "rb") as f:
            nb_meta = json.load(f)
        assert all(field in nb_meta for field in list(NotebookMetadata.model_fields))

        with open(os_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        nb["metadata"]["garden_metadata"] = nb_meta
        nbformat.write(nb, os_path, version=nbformat.NO_CONVERT)
