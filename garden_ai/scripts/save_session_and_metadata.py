# the contents of this script are appended to the user's automatically generated
# notebook script and run in the container in order to persist both a
# session.pkl and a metadata.json in the final image
if __name__ == "__main__":
    # save session after executing user notebook
    import dill  # type: ignore

    dill.dump_session("session.pkl")

    import json

    from pydantic.json import pydantic_encoder

    decorated_fns = []
    global_vars = list(globals().values())

    for obj in global_vars:
        if (
            hasattr(obj, "_pipeline_meta")
            and hasattr(obj, "_model_connectors")
            and hasattr(obj, "_garden_doi")
        ):
            decorated_fns.append(obj)

    if len(decorated_fns) == 0:
        raise ValueError("No functions marked with garden decorator.")

    total_meta = {}

    for marked in decorated_fns:
        key_name = marked.__name__
        connector_key = f"{key_name}.connectors"
        doi_key = f"{key_name}.garden_doi"

        total_meta[key_name] = marked._pipeline_meta
        # TODO add these as regular fields in PipelineMetadata/ RegisteredPipeline?
        total_meta[connector_key] = [
            connector.metadata for connector in marked._model_connectors
        ]
        if marked._garden_doi:
            total_meta[doi_key] = marked._garden_doi

    with open("metadata.json", "w+") as fout:
        json.dump(total_meta, fout, default=pydantic_encoder)
