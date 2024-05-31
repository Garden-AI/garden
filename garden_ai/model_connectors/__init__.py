from typing import Union, Optional

from pydantic import (
    HttpUrl,
    parse_obj_as,
    ValidationError,
)

from .exceptions import UnsupportedConnectorError
from .github_conn import GitHubConnector  # noqa: F401
from .hugging_face import HFConnector  # noqa: F401
from .model_connector import ModelConnector


def validate_repo_url(repo: Union[HttpUrl, str]) -> Optional[HttpUrl]:
    """Validate the given url.

    Returns: the URL if valid, otherwise None
    """
    try:
        return parse_obj_as(HttpUrl, repo)
    except ValidationError:
        return None


def create_connector(
    repo: Union[HttpUrl, str],
    repo_type: Optional[str] = None,
) -> ModelConnector:
    """Create a model connector to the given repo.

    Accepts either a full url:
    `con = create_connector("https://huggingface.co/Garden-AI/sklearn-iris")`

    or a repo identifier and repo type:
    `con = create_connector("Garden-AI/skelarn-iris", 'HF')`

    Args:
        repo (Union[HttpUrl, str]): The URL or identifier of the repository.
        repo_type (Optional[str]): The type of the repository ('GH' for GitHub, 'HF' for Hugging Face)

    Returns:
        ModelConnector: An instance of the appropriate connector class

    Raises:
        UnsupportedConnectorError: If the repository type is unable to be inferred,
            unspecified, or not a type we support.
    """

    # Try to determine the repo type based on the URL
    valid_url = validate_repo_url(repo)
    if valid_url:
        if "github.com" in valid_url.host:
            return GitHubConnector(repo_url=valid_url)
        if "huggingface.co" in valid_url.host:
            return HFConnector(repo_url=valid_url)
        else:
            raise UnsupportedConnectorError(
                f"We do not support repos from {valid_url.host}. Please use GitHub or Hugging Face."
            )
    # If repo is not a valid URL, assume repo is a repo_id in the form 'owner/repo'
    elif repo_type is not None:
        # determine repo type from user
        if repo_type == "GH":
            return GitHubConnector(repo_id=repo)
        elif repo_type == "HF":
            return HFConnector(repo_id=repo)
        else:
            raise UnsupportedConnectorError(
                "Unsupported repository type '{repo_type}'. Please use 'GH' for GitHub or 'HF' for Hugging Face"
            )
    else:
        raise UnsupportedConnectorError(
            "repo_type must be specified if repo is not a URL."
        )
