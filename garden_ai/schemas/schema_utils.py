from typing import Annotated, TypeAlias, TypeVar

from pydantic import AfterValidator, Field, HttpUrl, PlainSerializer, ValidationInfo
from pydantic_core import PydanticCustomError

T = TypeVar("T")

JsonStr: TypeAlias = str


# see: https://github.com/pydantic/pydantic-core/pull/820#issuecomment-1670475909
def _validate_unique_list(v: list[T]) -> list[T]:
    if len(v) != len(set(v)):
        raise PydanticCustomError("unique_list", "List must be unique")
    return v


def const_item_validator(cls, v: T, info: ValidationInfo) -> T:
    try:
        assert v == cls.model_fields[info.field_name].default
    except AssertionError:
        raise PydanticCustomError("const", "item is const")
    return v


# Types
UniqueList = Annotated[
    list[T],
    AfterValidator(_validate_unique_list),
    Field(json_schema_extra={"uniqueItems": True}, default_factory=list),
]
Url = Annotated[HttpUrl, PlainSerializer(lambda url: str(url), return_type=type(""))]
