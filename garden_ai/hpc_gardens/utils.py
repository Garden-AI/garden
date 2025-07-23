import io
from pathlib import Path

FIVE_MB = 5 * 1000 * 1000


def generate_xyz_str_chunks(xyz_file, max_size_bytes=FIVE_MB):
    """Generate string chunks that are under the max_size_bytes limit from the given xyz file

    Chunks are sets of complete xyz structures.
    """
    xyz_path = Path(xyz_file)
    if not xyz_path.exists():
        raise FileNotFoundError(xyz_file)

    chunk = io.StringIO()
    with open(xyz_path, "r") as f:
        while True:
            natoms_line = f.readline()
            if not natoms_line:
                # end of file
                break

            natoms = int(natoms_line.strip())
            comment_line = f.readline()

            structure = io.StringIO()
            structure.write(natoms_line)
            structure.write(comment_line)

            for _ in range(natoms):
                structure.write(f.readline())

            chunk_size = len(chunk.getvalue().encode("utf-8"))
            structure_size = len(structure.getvalue().encode("utf-8"))

            if (chunk_size + structure_size) > max_size_bytes:
                # the structure will push the chunk past the size limit, yield the chunk
                # then use the structure as the base for the next chunk
                yield chunk.getvalue()
                chunk = structure
            else:
                # the structure will fit in the chunk, add it and continue to the next structure
                chunk.write(structure.getvalue())

    # yield the final chunk if present
    if chunk.getvalue():
        yield chunk.getvalue()


def subproc_wrapper(func_source, *args, **kwargs):
    """Wrapper around a function to execute in a subprorcess using a conda env on the remote endpoint.

    This is designed to be serializable by globus-compute and get around python version
    mismatches and import errors that arise when the globus-compute endpoint is using
    a different python version and environment than the caller.
    """
    import subprocess
    import pickle
    import base64
    import tempfile
    import os
    import re

    # Extract function name from the source
    func_name_match = re.search(r"def\s+(\w+)\s*\(", func_source)
    if not func_name_match:
        return {"error": "Could not extract function name from source"}

    func_name = func_name_match.group(1)

    # Function data to execute
    func_data = {
        "source": func_source,
        "name": func_name,
        "args": args,
        "kwargs": kwargs,
    }

    # Encode function data
    encoded_data = base64.b64encode(pickle.dumps(func_data)).decode()

    # Python script to run in conda environment
    script = f"""import pickle
import base64

# Decode function data
func_data = pickle.loads(base64.b64decode("{encoded_data}"))

# Execute function source to define it
exec(func_data["source"])

# Get function object and execute it
func_obj = locals()[func_data["name"]]
result = func_obj(*func_data["args"], **func_data["kwargs"])

# Serialize and print result for capture
result_data = base64.b64encode(pickle.dumps(result)).decode()
print("RESULT_DATA:", result_data)
"""

    # Write script to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        # Run in template script in conda env
        cmd = [
            "conda",
            "run",
            "-p",
            "/home/hholb/.conda/envs/torch-sim-edith",
            "python",
            script_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        # Clean up temporary file
        os.unlink(script_path)

    if result.returncode != 0:
        return {"error": result.stderr, "stdout": result.stdout}

    # Extract result from subprocess output
    result_data = None

    for line in result.stdout.split("\n"):
        if line.startswith("RESULT_DATA: "):
            result_data = line[13:].strip()
            break

    if result_data is not None:
        return {
            "raw_data": result_data,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    return {
        "error": "No result found",
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def send_chunk_to_endpoint(chunk: str) -> str:
    import tempfile

    chunk_file = tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False)
    chunk_file.write(chunk)
    chunk_file.close()
    return chunk_file.name


def collate_file_chunks(master_file: str, chunk_file_names: list[str]) -> str:
    from pathlib import Path

    master_file = Path.home().joinpath(master_file)  # type: ignore[assignment]
    with open(master_file, "a") as f:
        for name in chunk_file_names:
            file = Path(name)
            assert file.exists()
            with open(file, "r") as chunk_f:
                f.writelines(chunk_f.readlines())
    return str(master_file)


def stream_result_chunk_from_file(result_file_path, chunk_index):
    """
    Read a specific chunk from a results file on the remote endpoint.
    Self-contained function with all necessary imports and logic inline.

    Args:
        result_file_path: Path to results file on remote endpoint
        chunk_index: Index of chunk to read (0-based)

    Returns:
        Dict with chunk_data, chunk_index, total_chunks, is_complete
    """
    from pathlib import Path
    import io

    FIVE_MB = 5 * 1000 * 1000

    def generate_result_file_chunks(result_file_path, max_size_bytes=FIVE_MB):
        """Generate string chunks from a results file - inline version."""
        result_path = Path(result_file_path)
        if not result_path.exists():
            raise FileNotFoundError(f"Results file not found: {result_path}")

        chunk = io.StringIO()
        with open(result_path, "r") as f:
            for line in f:
                # Check if adding this line would exceed the limit
                line_size = len(line.encode("utf-8"))
                chunk_size = len(chunk.getvalue().encode("utf-8"))

                if (chunk_size + line_size) > max_size_bytes and chunk_size > 0:
                    # Yield current chunk and start new one with this line
                    yield chunk.getvalue()
                    chunk = io.StringIO()
                    chunk.write(line)
                else:
                    # Add line to current chunk
                    chunk.write(line)

        # Yield final chunk if present
        if chunk.getvalue():
            yield chunk.getvalue()

    result_path = Path(result_file_path)
    if not result_path.exists():
        return {"error": f"Results file not found: {result_path}"}

    try:
        chunks = list(generate_result_file_chunks(result_path))
        total_chunks = len(chunks)

        if chunk_index >= total_chunks:
            return {
                "error": f"Chunk index {chunk_index} out of range (total: {total_chunks})"
            }

        return {
            "chunk_data": chunks[chunk_index],
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "is_complete": chunk_index == total_chunks - 1,
        }
    except Exception as e:
        return {"error": f"Failed to read results chunk: {str(e)}"}


def write_result_chunk_locally(chunk_data, local_file_path, mode="a"):
    """
    Write a chunk of results data to a local file.

    Args:
        chunk_data: String data to write
        local_file_path: Path to local output file
        mode: File open mode ('w' for first chunk, 'a' for subsequent)

    Returns:
        Success status
    """
    from pathlib import Path

    try:
        local_path = Path(local_file_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        with open(local_path, mode) as f:
            f.write(chunk_data)

        return {"success": f"Wrote chunk to {local_path}"}
    except Exception as e:
        return {"error": f"Failed to write chunk locally: {str(e)}"}
