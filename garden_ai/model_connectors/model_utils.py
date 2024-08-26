from typing import Type, Union

from pydantic import (
    HttpUrl,
)

from .github_conn import GitHubConnector
from .hugging_face import HFConnector
from .model_connector import ModelConnector


def _match_connector_type_by_url(url: str) -> Type[ModelConnector]:
    """Match url to the appropriate ModelRepository

    Args:
        url: A full HTTP URL as a str

    Returns:
        ModelConnector: The appropriate ModelConnector type for the URL.

    Raises:
        ValueError: When the URL is not able to be matched with a supported repository.
    """
    url = str(ModelConnector._is_valid_url(url))
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
    """Create a ModelConnector instance for a given repository URL.

    This function automatically determines the appropriate ModelConnector subclass based on the provided repository URL and instantiates it with the given arguments.

    Args:
        repo (HttpUrl | str): The URL of the repository. Currently supports GitHub and Hugging Face repositories.
        **kwargs: Additional keyword arguments passed directly to the ModelConnector's `__init__` method. These can include options like 'branch', 'revision', 'local_dir', etc. See [ModelConnector][garden_ai.model_connectors.ModelConnector] attributes for all available options.

    Returns:
        ModelConnector: An instance of the appropriate ModelConnector subclass (e.g., GitHubConnector, HFConnector) for the given repository.

    Raises:
        ValueError: If the repository type is not supported or the URL is invalid.

    Examples:
        ```python
            my_connector = create_connector("https://huggingface.co/my-model", branch="dev", revision="abc123")

            @entrypoint(metadata=..., model_connectors=[my_connector])
            def my_entrypoint(data):
                # calls `.stage()` method to fetch repository contents
                dowload_path = my_connector.stage()
                # do something with the downloaded weights
                ...
                return result
        ```

    Note:
        The function infers the connector type from the URL. Make sure to provide a valid URL for the repository you want to connect to.
    """  # noqa: E501
    matched_connector = _match_connector_type_by_url(str(repo))
    return matched_connector(repo_url=str(repo), **kwargs)  # type: ignore[misc]
