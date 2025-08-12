import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from garden_ai import GardenClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("garden-mcp-server")


# MLIP-specific tools
try:
    from .mlip import check_job_status, get_job_results, submit_relaxation_job

    # only include mlip tools if mlip extra is also installed
    mcp.tool()(submit_relaxation_job)
    mcp.tool()(check_job_status)
    mcp.tool()(get_job_results)

except ImportError:
    pass


@mcp.tool()
def search_gardens(
    query: str = "",
    limit: int = 10,
    offset: int = 0,
) -> str:
    """
    Search gardens using full-text search across titles, descriptions, and other metadata.

    Args:
        query: Search terms to find in garden metadata. Searches across titles, descriptions,
               authors, and other text fields. Use natural language or keywords.
        limit: Maximum number of results to return
        offset: Number of results to skip for pagination.

    Returns:
        JSON string containing search results with pagination info:
        - count: Number of results in this response
        - total: Total number of matching gardens
        - offset: Current pagination offset
        - gardens: List of matching gardens with doi, title, description, tags, and functions
    """
    archived_filter = {"field_name": "is_archived", "values": ["false"]}

    search_payload = {
        "q": query,
        "limit": limit,
        "offset": offset,
        "filters": [archived_filter],
    }

    try:
        client = GardenClient().backend_client
        response = client.search_gardens(search_payload)

        filtered_response = {
            "count": response.get("count"),
            "total": response.get("total"),
            "offset": response.get("offset"),
        }

        filtered_gardens = []
        for garden in response["garden_meta"]:
            garden_md = {k: garden[k] for k in ["doi", "title", "description", "tags"]}

            functions = garden["modal_functions"]
            function_metadata = [
                {k: f[k] for k in ["function_name", "description"]} for f in functions
            ]

            # Combine garden metadata with function names
            garden_with_functions = {**garden_md, "functions": function_metadata}

            filtered_gardens.append(garden_with_functions)

        filtered_response["gardens"] = filtered_gardens

        return json.dumps(filtered_response, default=str)

    except Exception as e:
        raise ToolError(f"Error searching for garden: {e}")


@mcp.tool()
def get_garden_metadata(doi: str) -> str:
    """
    Fully describe a garden with a given doi. In addition to other useful metadata,
    this contains the list of functions callable from the garden.

    If the user wants you to call one of these functions, use the `generate_script` tool to get the metadata needed to generate a script
    which calls the function, then use the `run_script` tool to execute the script.
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


@mcp.tool()
def generate_script(doi: str, function_name: str):
    """
    Get metadata needed to generate a concise Python script which calls a Garden function. Use this tool before using the `run_script` tool to
    execute the script.

    This tool retrieves metadata including minimal usage examples and original source code of a function in a Garden.

    <script specification>
        - The script you generate will be executed on behalf of the user with the `run_script` tool.
        - The script is designed to be minimal and focused on core functionality, providing users with clean, executable code examples.
        - Minimize or omit any additional code in the script, such as print statements, error handling, or additional explanation.
        - The script should be executable as is, without any additional code.
        - If the user has provided data to pass to the function, the script should load and/or preprocess the data if needed.
    </script specification>

    Args:
        doi: Digital Object Identifier of the garden
        function_name: Name of the function to generate the script for. Can be either a direct
                        function name or "ClassName.method_name" format

    Returns:
        dict: Function metadata including:
                - Basic implementation pattern following:
                    ```python
                    from garden_ai import GardenClient
                    client = GardenClient()
                    my_garden = client.get_garden(doi)
                    my_garden.my_function(my_params)
                    ```
                - Instructions for minimal usage examples only (no print statements, no error handling, no additional explanation)
                - Function text for context, to better infer information like function arguments and return type.

    Raises:
        ValueError: If the function name is not found in the specified garden
    """
    garden = GardenClient().backend_client.get_garden(doi)

    if "." in function_name:
        cls_name, fn_name = function_name.split(".")
        modal_cls = getattr(garden, cls_name)
        modal_fn = getattr(modal_cls, fn_name)
    else:
        modal_fn = getattr(garden, function_name)

    if modal_fn is None:
        raise ValueError("Function id is not valid")

    response = {
        "response_type": "garden_client_code_generation",
        "imports": "from garden_ai import GardenClient",
        "function_metadata": {
            "function_signature": f"my_garden.{modal_fn.metadata.function_name}",
            "doi": doi,
            "description": modal_fn.metadata.description,
            "function_text": modal_fn.metadata.function_text,
        },
    }

    return response


@mcp.tool()
def run_script(script: str):
    """
    Execute a Python script and return the result.
    Do not use this tool without first using the `generate_script` tool to fetch accurate context needed to call the function.

    This tool runs Python scripts generated by the `generate_script` tool. The script MUST store its
    final output in a variable named `result`. The final output MUST be JSON-serializable.

    Args:
        script: Python code string to execute. The script MUST define a variable called `result` containing the final answer/output.

    Returns:
        The value stored in the 'result' variable (must be JSON-serializable)

    Example usage:
        Script should look something like:
        ```python
        from garden_ai import GardenClient
        client = GardenClient()
        my_garden = client.get_garden("10.23677/example")

        # optionally, load and/or preprocess input data if needed
        # ...
        output = my_garden.some_function(my_input_data)

        # optionally postprocess output if needed
        # ...
        result = output
        ```

    Raises:
        ToolError: If script execution fails, if no 'result' variable is defined, or if the result is not JSON-serializable
    """
    global_vars: dict[str, Any] = {}

    try:
        exec(script, global_vars)
    except Exception as e:
        raise ToolError(f"Script execution failed: {str(e)}")

    if "result" not in global_vars:
        raise ToolError(
            "Script must store final answer in 'result'. Example: result = 42"
        )

    result = global_vars["result"]

    try:
        json.dumps(result)
    except (TypeError, ValueError) as e:
        raise ToolError(f"Result is not JSON-serializable: {str(e)}")

    return result


def main():
    """Run the MCP server."""
    mcp.run(transport="stdio")
