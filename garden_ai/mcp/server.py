import logging

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


# copied from globus-labs/science-mpcs repo
@mcp.tool()
def get_functions(doi: str):
    """
    Return a list of available modal function names for this Garden.
    """
    garden = GardenClient().get_garden(doi)
    data = garden.metadata.model_dump(
        exclude={"owner_identity_id", "id", "language", "publisher"}
    )
    data["entrypoints"] = [ep.metadata.model_dump() for ep in garden.entrypoints]
    data["modal_functions"] = [
        mf.metadata.model_dump() for mf in garden.modal_functions
    ]
    return [f["function_name"] for f in data["modal_functions"]]


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
    entrypoint = getattr(garden, func_name, None)
    if entrypoint is None:
        raise ToolError(f"No such function '{func_name}' in garden '{doi}'")
    try:
        result = entrypoint(func_args)
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


def main():
    """Run the MCP server."""
    mcp.run(transport="stdio")
