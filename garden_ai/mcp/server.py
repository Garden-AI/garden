import logging
import json

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from garden_ai import GardenClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("garden-mcp-server")


@mcp.tool()
def echo(message: str) -> str:
    """
    Echo a message back, while sanity checking that GardenClient auth works.

    Args:
        message: The message to echo back

    Returns:
        The echoed message with authentication status
    """
    try:
        client = GardenClient()
        user_email = client.get_email()
        logger.info(f"Garden authentication successful for user: {user_email}")

        return f"Echo from Garden (logged in as {user_email}): {message}"
    except Exception as e:
        logger.error(f"Garden authentication failed: {e}")
        return f"Garden auth failed: {e}"


@mcp.tool()
def get_functions(doi: str):
    """
    Return a list of available modal function names for this Garden.
    """
    garden = GardenClient().get_garden(doi)
    data = garden.metadata.model_dump(
        exclude={"owner_identity_id", "id", "language", "publisher"}
    )

    if garden.modal_functions:
        data["modal_functions"] = [
            mf.metadata.model_dump() for mf in garden.modal_functions
        ]
    elif garden.modal_classes:
        data["modal_functions"] = []
        for modal_class in garden.modal_classes:
            for method in modal_class._methods.values():
                data["modal_functions"].append(method.metadata.model_dump())

    return data["modal_functions"]


@mcp.tool()
def run_function(
    doi: str,
    func_name: str,
    func_args: list[str],
):
    """
    Load the Garden by DOI, locate the named function, and invoke it with the provided args.
    """
    garden = GardenClient().get_garden(doi)

    if garden.modal_functions:
        function = getattr(garden, func_name, None)
    if garden.modal_classes:
        class_name, class_method = func_name.split(".")
        modal_class = getattr(garden, class_name)
        function = getattr(modal_class, class_method)

    if function is None:
        raise ToolError(f"No such function '{func_name}' in garden '{doi}'")
    try:
        result = function(func_args)
    except Exception as e:
        raise ToolError(f"Error running '{func_name}(...)': {e}")
    return result


# MLIP-specific tools

try:
    import ase  # noqa
    from .mlip import submit_relaxation_job, check_job_status, get_job_results

    # only include mlip tools if mlip extra is also installed
    mcp.tool()(submit_relaxation_job)
    mcp.tool()(check_job_status)
    mcp.tool()(get_job_results)

except ImportError:
    pass


@mcp.tool()
def search_gardens(
    dois: list[str] | None = None,
    tags: list[str] | None = None,
    draft: bool | None = None,
    authors: list[str] | None = None,
    contributors: list[str] | None = None,
    year: str | None = None,
    owner_uuid: str | None = None,
    limit: int = 20,
) -> str:
    """
    Search for gardens based on inputs, returns at most limit gardens.
    """
    try:
        response = GardenClient().backend_client.get_gardens(
            dois=dois,
            tags=tags,
            draft=draft,
            authors=authors,
            contributors=contributors,
            year=year,
            owner_uuid=owner_uuid,
            limit=limit,
        )

        result = []
        for garden in response:
            data = garden.metadata.model_dump(
                mode="json", include={"doi", "title", "description", "tags"}
            )

            if garden.modal_functions:
                data["modal_functions"] = [
                    mf.metadata.function_name for mf in garden.modal_functions
                ]
            elif garden.modal_classes:
                data["modal_functions"] = []
                for modal_class in garden.modal_classes:
                    for method in modal_class._methods.values():
                        data["modal_functions"].append(method.metadata.function_name)

            result.append(data)

        return json.dumps(result)
    except Exception as e:
        raise ToolError(f"Error searching for garden: {e}")


@mcp.tool()
def get_garden_metadata(doi: str) -> str:
    """
    Fully describe garden by doi
    """
    try:
        response = GardenClient().backend_client.get_garden(doi)
        metadata = response.metadata.model_dump(mode="json")

        if response.modal_functions:
            metadata["modal_functions"] = [
                mf.metadata.function_name for mf in response.modal_functions
            ]
        elif response.modal_classes:
            metadata["modal_functions"] = []
            for modal_class in response.modal_classes:
                for method in modal_class._methods.values():
                    metadata["modal_functions"].append(method.metadata.function_name)
        else:
            metadata["modal_functions"] = []

        return json.dumps(metadata)
    except Exception as e:
        raise ToolError(f"Error getting garden metadata: {e}")


def main():
    """Run the MCP server."""
    mcp.run(transport="stdio")
