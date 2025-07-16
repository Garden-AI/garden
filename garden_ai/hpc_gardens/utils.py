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
