import json

MAX_BYTES = 5 * 1024 * 1024


def is_over_size_limit(notebook_json: dict) -> bool:
    """
    Return True if notebook contents is over 5MB.
    """
    # Convert to byte string to account for longer unicode characters
    as_byte_string = json.dumps(notebook_json).encode("utf-8")
    return len(as_byte_string) > MAX_BYTES


def clear_cells(notebook_json: dict) -> dict:
    """
    Returns new notebook with all cell outputs cleared.
    """
    new_nb = notebook_json.copy()
    for cell in new_nb["cells"]:
        if "outputs" in cell:
            cell["outputs"] = []
    return new_nb
