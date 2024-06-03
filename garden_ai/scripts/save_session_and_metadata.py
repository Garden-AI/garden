# the contents of this script are appended to the user's automatically generated
# notebook script and run in the container in order to persist both a
# session.pkl and a metadata.json in the final image


def assert_compatible_dill_version():
    """sanity check that dill version is correct w/r/t the python version.

    This is factored out into its own function because `sys.version_info` does
    something spooky that breaks dill if it picks up the reference.
    """
    from importlib.metadata import version
    import sys

    # this should be exactly the same as globus compute requirements
    python_version = sys.version_info
    dill_version = version("dill")
    required_dill_version = "0.3.5.1" if python_version < (3, 11) else "0.3.6"

    if dill_version != required_dill_version:
        message = (
            "To ensure compatibility with Globus Compute endpoints, Garden needs "
            f"dill version {required_dill_version} when running python version "
            f"{'.'.join(map(str, python_version[:2]))}. However, this environment "
            f"has dill version {dill_version} installed, possibly due to other "
            "packages installed from within your notebook. \n\nIf you are installing "
            "dependencies from a cell of your notebook, try specifying them in a "
            "requirements file with the `--requirements` flag from the command line "
            "instead and try again. "
        )
        raise EnvironmentError(message)


def get_requirements_file():
    """Return pathlib.Path to a requirements file in the container if one exists.

    Requirements are written into the container in the same directory as the notebook.
    See garden_ai/app/notebook.py `publish`.
    """
    from pathlib import Path

    pip = Path(__file__).parent / "requirements.txt"
    if pip.exists():
        return pip

    conda = Path(__file__).parent / "requirements.yml"
    if conda.exists():
        return conda

    return None


def get_requirements_data():
    """Return a list of requirements for the current notebook."""
    from garden_ai.notebook_metadata import read_requirements_data

    path = get_requirements_file()
    if not path:
        return None

    reqs = read_requirements_data(path)
    return reqs.contents if reqs else None


if __name__ == "__main__":
    import dill  # type: ignore

    assert_compatible_dill_version()

    # save session after executing user notebook
    dill.dump_session("session.pkl")

    import json
    import os

    from pydantic_core import to_jsonable_python
    from garden_ai.model_connectors import HFConnector, GitHubConnector

    entrypoint_fns, step_fns, steps = [], [], []
    global_vars = list(globals().values())
    global_notebook_doi = os.environ.get("GLOBAL_NOTEBOOK_DOI", None)

    for obj in global_vars:
        if hasattr(obj, "_garden_entrypoint"):
            entrypoint_fns.append(obj)

        if hasattr(obj, "_garden_step"):
            step_fns.append(obj)

        if isinstance(obj, (HFConnector, GitHubConnector)):
            if obj.stage.has_been_called:
                raise RuntimeWarning(
                    f"{obj}'s `.stage()` method was called unexpectedly during "
                    "the build process. Double check that no top-level code "
                    "calls your entrypoint in the final version of your notebook. "
                )

    if len(entrypoint_fns) == 0:
        raise ValueError("No functions marked with garden_entrypoint decorator.")

    requirements_data = get_requirements_data()

    total_meta = {}

    for entrypoint_fn in entrypoint_fns:
        key_name = entrypoint_fn.__name__
        doi_key = f"{key_name}.garden_doi"
        step_key = f"{key_name}.entrypoint_step"
        entrypoint_meta = entrypoint_fn._garden_entrypoint

        total_meta[key_name] = entrypoint_meta.dict()
        total_meta[key_name]["test_functions"] = entrypoint_meta._test_functions
        if entrypoint_meta._target_garden_doi:
            total_meta[doi_key] = entrypoint_meta._target_garden_doi
        elif global_notebook_doi:
            total_meta[doi_key] = global_notebook_doi
        else:
            raise ValueError(
                f"Entrypoint {key_name} has no DOI associated with it. "
                "Either provide a global notebook DOI in your notebook metadata or provide the entrypoint decorator with a DOI."
            )
        total_meta[step_key] = entrypoint_meta._as_step
        total_meta[key_name]["requirements"] = requirements_data

    for step_fn in step_fns:
        # Relying on insertion order being maintained in dicts in Python 3.8 forward 🤠
        steps.append(step_fn._garden_step.dict())

    total_meta["steps"] = steps

    with open("metadata.json", "w+") as fout:
        json.dump(total_meta, fout, default=to_jsonable_python)
