# the contents of this script are appended to the user's automatically generated
# notebook script and run in the container in order to persist both a
# session.pkl and a metadata.json in the final image
if __name__ == "__main__":
    # save session after executing user notebook
    import dill  # type: ignore

    dill.dump_session("session.pkl")

    import json

    from pydantic.json import pydantic_encoder

    pipeline_fns, step_fns, steps = [], [], []
    global_vars = list(globals().values())

    for obj in global_vars:
        if hasattr(obj, "_garden_pipeline"):
            pipeline_fns.append(obj)

        if hasattr(obj, "_garden_step"):
            step_fns.append(obj)

    if len(pipeline_fns) == 0:
        raise ValueError("No functions marked with garden_pipeline decorator.")

    total_meta = {}

    for pipeline_fn in pipeline_fns:
        key_name = pipeline_fn.__name__
        doi_key = f"{key_name}.garden_doi"
        step_key = f"{key_name}.pipeline_step"
        pipeline_meta = pipeline_fn._garden_pipeline

        total_meta[key_name] = pipeline_meta.dict()
        if pipeline_meta._target_garden_doi:
            total_meta[doi_key] = pipeline_meta._target_garden_doi
        total_meta[step_key] = pipeline_meta._as_step

    for step_fn in step_fns:
        # Relying on insertion order being maintained in dicts in Python 3.8 forward 🤠
        steps.append(step_fn._garden_step.dict())

    total_meta["steps"] = steps

    with open("metadata.json", "w+") as fout:
        json.dump(total_meta, fout, default=pydantic_encoder)
