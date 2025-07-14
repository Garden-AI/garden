import logging

from mcp.server.fastmcp import FastMCP

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


def main():
    """Run the MCP server."""
    mcp.run(transport="stdio")
