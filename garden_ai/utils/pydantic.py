from pydantic import ValidationInfo
from pydantic_core import PydanticCustomError
from typing import Optional, TypeVar

T = TypeVar("T")


def const_item_validator(cls, v: Optional[T], info: ValidationInfo) -> Optional[T]:
    try:
        assert v == cls.model_fields[info.field_name].default
    except AssertionError:
        raise PydanticCustomError("const", "item is const")
    return v
