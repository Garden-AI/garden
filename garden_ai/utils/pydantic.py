from pydantic_core import PydanticCustomError


def unique_items_validator(v):
    if v is None:
        return v
    try:
        if len(v) != len(set(v)):
            raise PydanticCustomError("unique_list", "list must be unique")
        return v
    except TypeError:
        pass
    if len(v) != len(set([item.model_dump_json() for item in v])):
        raise PydanticCustomError("unique_list", "list must be unique")
    return v
