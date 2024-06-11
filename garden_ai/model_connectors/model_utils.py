from typing import Type, Union, Optional

from pydantic import (
    HttpUrl,
)

from .github_conn import GitHubConnector
from .hugging_face import HFConnector
from .model_connector import ModelConnector


def _match_connector_type_by_url(url: str) -> Optional[Type[ModelConnector]]:
    """Match url to the appropriate ModelRepository

    Args:
        url: A full HTTP URL as a str

    Returns:
        ModelConnector: The appropriate ModelConnector type for the URL.

    Raises:
        ValueError: When the URL is not able to be matched with a supported repository.
    """
    url = str(ModelConnector.is_valid_url(url))
    if url:
        if "github.com" in url:
            return GitHubConnector
        elif "huggingface.co" in url:
            return HFConnector
        else:
            raise ValueError("Repository type is not supported.")
    else:
        raise ValueError("Invalid URL")


def create_connector(
    repo: Union[HttpUrl, str],
    **kwargs,
) -> ModelConnector:
    """Create a model connector to the given repo.

    Accepts a full url:
    `con = create_connector("https://huggingface.co/Garden-AI/sklearn-iris")`
    `con = create_connector("https://github.com/Garden-AI/garden)`

    Args:
        repo (Union[HttpUrl, str]): The URL of the repository.
        **kwargs: any other keyword arguments are passed directly to the ModelConnector's __init__
            see `garden_ai.model_connectors.model_connector.ModelConnector` for specifics.

    Returns:
        ModelConnector: An instance of the appropriate connector class
    """
    matched_connector = _match_connector_type_by_url(str(repo))
    return matched_connector(repo_url=str(repo), **kwargs)  # type: ignore[misc]
