import json
import logging
from pathlib import Path

from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from garden_ai import GardenClient, get_garden

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("garden-mcp-server")


@mcp.tool()
def invoke_function(
    garden_doi: str,
    function_name: str,
    input_data_or_path: str,
):
    """
    Invoke a function from a Garden with flexible input handling.

    This tool executes functions from Gardens (computational research artifacts) with intelligent
    input processing and comprehensive error handling.

    Args:
        garden_doi: Digital Object Identifier of the garden (e.g., "10.23677/m0dr-jg09")
        function_name: Name of the function to invoke. Can be either a direct function name
                      (e.g., "my_function") or a class method (e.g., "ClassName.method_name")
        input_data_or_path: Input data for the function. This parameter is highly flexible:
                           - If it's a valid file path, the file contents will be read and used
                           - If it's valid JSON, it will be parsed as structured data
                           - Otherwise, it will be passed as a plain string

    Returns:
        The function's output, which must be JSON-serializable

    Usage Examples:
        - String input: invoke_function("10.23677/example", "process_text", "hello world")
        - JSON input: invoke_function("10.23677/example", "analyze_data", '{"key": "value"}')
        - File input: invoke_function("10.23677/example", "analyze_file_contents", "/path/to/data.txt")
        - Class method: invoke_function("10.23677/example", "Model.predict", "input_data")

    Note: If the function returns non-JSON-serializable objects, or requires non-JSON-serializable inputs,
    use the generate_script tool to create code for the user to manually invoke.
    """
    garden = get_garden(garden_doi)
    # verify function exists on this garden
    try:
        if "." in function_name:
            cls_name, fn_name = function_name.split(".")
            modal_cls = getattr(garden, cls_name)
            modal_fn = getattr(modal_cls, fn_name)
        else:
            modal_fn = getattr(garden, function_name)
    except AttributeError:
        existing_names = [fn.metadata.function_name for fn in garden.modal_functions]
        existing_names += [
            fn.metadata.function_name
            for cls in garden.modal_classes
            for fn in cls._methods.values()
        ]
        if existing_names:
            msg_extra = f"\nDid you mean: {', '.join(set(existing_names))}?"
        else:
            msg_extra = ""
        raise ToolError(
            f"Garden {garden_doi} does not have a function {function_name}." + msg_extra
        )

    # try parsing input str as path to file
    if (path := Path(input_data_or_path).resolve()).exists():
        input_data = path.read_text()
    else:
        input_data = input_data_or_path

    # try parsing input str (or file contents) as json
    try:
        input_data = json.loads(input_data)
        results = modal_fn(input_data)
    except json.JSONDecodeError:
        # not json, try as a plain str
        results = modal_fn(input_data)
    except Exception as e:
        raise ToolError(
            "Function invocation failed. "
            "Try generating sample code with the "
            "`generate_script` tool and invoking manually to troubleshoot. "
            f"Error: {e}"
        ) from e
    # if the result object is not a jsonable type we need to handle it here or
    # it just crashes the MCP server without an error message
    try:
        _ = json.dumps(results)
        return results
    except TypeError as e:
        if "JSON" in str(e):
            import sys

            print(f"Error: could not serialize results to JSON: {e}", file=sys.stderr)
            raise ToolError(
                f"Error: {e}\n"
                "Try generating sample code with the `generate_script` tool and invoking the function manually. "
            ) from e
        else:
            raise


# MLIP-specific tools

try:
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
            if garden.metadata.state == "ARCHIVED":
                continue
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
    Fully describe garden by doi. In addition to other informative metadata,
    this contains the list of functions callable from the garden.
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
    Generate a concise Python script for using a function from Garden

    This tool retrieves metadata and generates minimal usage examples for functions from Garden.
    It provides the essential code structure needed to call a specific function from Garden.

    Args:
        doi: Digital Object Identifier of the garden
        function_name: Name of the function to generate the script for. Can be either a direct
                        function name or "ClassName.method_name" format

    Returns:
        dict: Function metadata including:
                - Basic implementation pattern following:
                    '''python
                    from garden_ai import GardenClient
                    client = GardenClient()
                    my_garden = client.get_garden(doi)
                    my_garden.my_function(my_params)
                    '''
                - Instructions for minimal usage examples only (no print statements, no error handling, no additional explanation)
                - Function text for client to parse an get relevant information like function arguments and return type.

    The generated scripts are designed to be concise and focused on core functionality providing users with clean, executable code examples

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

    This tool runs Python scripts, typically those generated by the generate_script tool. The script must store its
    final output in a variable named 'result'.

    Args:
        script: Python code string to execute. The script MUST define a variable called 'result' containing the final answer/output.

    Returns:
        The value stored in the 'result' variable (must be JSON-serializable)

    Example usage:
        Script should end with something like:
        '''
        from garden_ai import GardenClient
        client = GardenClient()
        my_garden = client.get_garden("10.23677/example")
        output = my_garden.some_function(parameters)

        result = output
        ```

    Raises:
        ToolError: If script execution failes, if no 'result' variable is defined, or if the result is not JSON-serializable
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
