# the contents of this script are appended to the automatically generated
# notebook script and run in the container in order to persist both a
# session.pkl and a metadata.json in the final image
if __name__ == "__main__":
    # save session after executing user notebook
    import dill  # type: ignore

    dill.dump_session("session.pkl")

    from pydantic.json import pydantic_encoder

    import json

    decorated_fns = []
    global_vars = list(globals().values())

    for obj in global_vars:
        if hasattr(obj, "_pipeline_meta") and hasattr(obj, "_model_connectors"):
            decorated_fns.append(obj)

    if len(decorated_fns) == 0:
        raise ValueError("No functions marked with garden decorator.")

    total_meta = {}

    for marked in decorated_fns:
        key_name = marked._pipeline_meta["short_name"]
        connector_key = f"{key_name}.connectors"

        total_meta[key_name] = marked._pipeline_meta
        total_meta[connector_key] = [
            connector.metadata for connector in marked._model_connectors
        ]

    with open("metadata.json", "w+") as fout:
        json.dump(total_meta, fout, default=pydantic_encoder)
