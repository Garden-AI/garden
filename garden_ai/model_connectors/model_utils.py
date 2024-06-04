import re
from typing import Dict, Type, Union, Optional

from pydantic import (
    HttpUrl,
    parse_obj_as,
    ValidationError,
)

from .exceptions import (
    ConnectorInvalidUrlError,
    ConnectorInvalidRepoIdError,
    UnsupportedConnectorError,
)
from .github_conn import GitHubConnector
from .hugging_face import HFConnector
from .model_connector import ModelConnector, ModelRepository


# Maps ModelRepositorys to their corresponding connector class.
CONNECTOR_MAPPING: Dict[ModelRepository, Type[ModelConnector]] = {
    ModelRepository.GITHUB: GitHubConnector,
    ModelRepository.HUGGING_FACE: HFConnector,
}


def match_repo_type_by_url(url: str) -> ModelRepository:
    """Match url to the appropriate ModelRepository

    Args:
        url: A full HTTP URL as a str

    Returns:
        ModelConnector: The appropriate ModelConnector type for the URL.

    Raises:
        ValueError: When the URL is not able to be matched with a supported repository.
    """
    if is_valid_url(url):
        if "github.com" in url:
            return CONNECTOR_MAPPING.get(ModelRepository.GITHUB)
        elif "huggingface.co" in url:
            return CONNECTOR_MAPPING.get(ModelRepository.HUGGING_FACE)
        else:
            raise ValueError("Repository type is not supported.")


def match_repo_type_by_id_and_type(repo_id: str, repo_type: str) -> ModelRepository:
    """Match repo_id to the appropriate ModelRepository

    Args:
        repo_id (str): A repository id in the form 'owner/repo'
        repo_type (str): A repository type, e.g 'GH' for GitHub, 'HF' for Hugging Face

    Raises:
        ValueError: When the the given repo_type is unsupported.
    """
    if is_valid_repo_id(repo_id):
        match repo_type:
            case "GH":
                return CONNECTOR_MAPPING.get(ModelRepository.GITHUB)
            case "HF":
                return CONNECTOR_MAPPING.get(ModelRepository.HUGGING_FACE)
            case _:
                raise ValueError("Unsupported repo_type")


def is_valid_url(repo: Union[HttpUrl, str]) -> Optional[HttpUrl]:
    """Validate the given url.

    Returns: the URL if valid, otherwise None

    Raises:
       ConnectorInvalidUrlError: When the url is not valid.
    """
    try:
        return parse_obj_as(HttpUrl, repo)
    except ValidationError:
        raise ConnectorInvalidUrlError(
            f"Invalid repo url: {repo}. Repo url must be a valid HTTP URL."
        )


def is_valid_repo_id(repo_id: str) -> Optional[str]:
    """Parse repo_id to make sure it is in the form 'owner/repo'

    Return: repo_id as a string

    Raises:
        ValidationError: When repo_id is not in the form 'owner/repo'
    """
    if re.fullmatch(r"^[a-zA-Z0-9-]+\/[a-zA-Z0-9-]+$", repo_id.strip()):
        return repo_id
    else:
        raise ConnectorInvalidRepoIdError(
            f"Invalid repo_id: {repo_id}. Must be in the form 'owner/repo'"
        )


def create_connector(
    repo: Union[HttpUrl, str],
    **kwargs,
) -> ModelConnector:
    """Create a model connector to the given repo.

    Accepts either a full url:
    `con = create_connector("https://huggingface.co/Garden-AI/sklearn-iris")`

    or a repo identifier and repo type:
    `con = create_connector("Garden-AI/skelarn-iris", repo_type='HF')`

    Args:
        repo (Union[HttpUrl, str]): The URL or identifier of the repository.
        **kwargs: any other keyword arguments are passed directly to the ModelConnector's __init__
            see `garden_ai.model_connectors.model_connector.ModelConnector` for specifics.

    Returns:
        ModelConnector: An instance of the appropriate connector class

    Raises:
        UnsupportedConnectorError: If the repository type is unable to be inferred or not a type we support.
    """
    try:
        # Try and match the repo type by URL, throws error
        matched_repo = match_repo_type_by_url(repo)
        return matched_repo(repo_url=repo, **kwargs)
    except ConnectorInvalidUrlError:
        try:
            # Try and match the repo type by id and type
            repo_type = kwargs.get("repo_type")
            if repo_type is not None:
                matched_repo = match_repo_type_by_id_and_type(repo, repo_type)
                return matched_repo(repo_id=repo, **kwargs)
            else:
                raise UnsupportedConnectorError(
                    "Unable to create ModelConnector.\n"
                    "Excpected ether URL or repo_id and repo_type.\n"
                    "Try:\n"
                    "\tc = create_connector('https:github.com/owner/repo')\n"
                    "or:\n"
                    "\tc = create_connector('owner/repo', 'GH')"
                )
        except Exception as e:
            raise e
