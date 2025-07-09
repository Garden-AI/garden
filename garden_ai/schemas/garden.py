from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field

from .schema_utils import UniqueList


class GardenMetadata(BaseModel):
    """Represents the metadata defining a Garden.

    **Not meant to be instantiated directly by users.** Instead, new Gardens should be created via the CLI or [web UI](https://thegardens.ai/#/garden/create)

    Attributes:
        doi (str): The Digital Object Identifier (DOI) for the Garden.
        title (str): The title of the Garden.
        authors (UniqueList[str]): A list of authors of the Garden.
        contributors (UniqueList[str]): A list of contributors to the Garden. Defaults to an empty list.
        doi_is_draft (bool | None): Indicates if the DOI is in draft status. Defaults to True.
        description (str | None): A brief description of the Garden. Defaults to None.
        publisher (str): The publisher of the Garden. Defaults to "Garden-AI".
        year (str): The year of publication. Defaults to the current year.
        language (str): The primary language of the Garden. Defaults to "en" (English).
        tags (UniqueList[str]): A list of tags associated with the Garden. Defaults to an empty list.
        version (str): The version of the Garden. Defaults to "0.0.1".
        owner_identity_id (UUID | None): The UUID of the Garden's owner. Defaults to None.
        id (int | None): An internal identifier for the Garden. Defaults to None.
    """  # noqa: E501

    doi: str
    title: str
    authors: UniqueList[str]

    contributors: UniqueList[str] = Field(default_factory=list)
    doi_is_draft: bool | None = True
    description: str | None = None
    publisher: str = "Garden-AI"
    year: str = Field(default_factory=lambda: str(datetime.now().year))
    language: str = "en"
    tags: UniqueList[str] = Field(default_factory=list)
    version: str = "0.0.1"

    # these are DB ids (unlike entrypoint_ids which are dois)
    modal_function_ids: UniqueList[int] = Field(default_factory=list)

    owner_identity_id: UUID | None = None
    id: int | None = None
