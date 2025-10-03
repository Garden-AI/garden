import random
from pathlib import Path

import pytest

from garden_ai.hpc.utils import check_file_size_and_read


@pytest.fixture
def small_xyz_file(tmpdir):
    """Create a small XYZ file that's under the 5MB limit."""
    symbols = ["Cu", "Si", "Pb"]
    nstructures = 10
    natoms = 5

    tmpfile = tmpdir.join("small_test.xyz")
    with open(tmpfile, "w") as f:
        for _ in range(nstructures):
            f.write(f"{natoms}\n")
            f.write("Comment Line\n")
            for _ in range(natoms):
                symbol = random.choice(symbols)
                positions = [random.random() for _ in range(3)]
                f.write(
                    f"{symbol} {positions[0]:.2f} {positions[1]:.2f} {positions[2]:.2f}\n"
                )
    yield tmpfile


@pytest.fixture
def large_xyz_file(tmpdir):
    """Create a large XYZ file that exceeds the 5MB limit."""
    symbols = ["Cu", "Si", "Pb"]
    nstructures = 10000
    natoms = 100

    tmpfile = tmpdir.join("large_test.xyz")
    with open(tmpfile, "w") as f:
        for _ in range(nstructures):
            f.write(f"{natoms}\n")
            f.write("Comment Line\n")
            for _ in range(natoms):
                symbol = random.choice(symbols)
                positions = [random.random() for _ in range(3)]
                f.write(
                    f"{symbol} {positions[0]:.2f} {positions[1]:.2f} {positions[2]:.2f}\n"
                )
    yield tmpfile


def test_check_file_size_and_read_small_file(small_xyz_file):
    """Test that small files are read successfully."""
    content = check_file_size_and_read(small_xyz_file)
    assert isinstance(content, str)
    assert len(content) > 0
    # Should contain the expected structure
    assert "Comment Line" in content


def test_check_file_size_and_read_large_file_raises_error(large_xyz_file):
    """Test that large files raise a ValueError."""
    with pytest.raises(ValueError, match="File size .* exceeds maximum allowed size"):
        check_file_size_and_read(large_xyz_file)


def test_check_file_size_and_read_nonexistent_file(tmpdir):
    """Test that nonexistent files raise FileNotFoundError."""
    nonexistent_file = tmpdir.join("nonexistent.xyz")

    with pytest.raises(FileNotFoundError, match="XYZ file not found"):
        check_file_size_and_read(nonexistent_file)


def test_check_file_size_and_read_empty_file(tmpdir):
    """Test that empty files are handled correctly."""
    empty_file = tmpdir.join("empty.xyz")
    Path(empty_file).touch()  # Create empty file

    content = check_file_size_and_read(empty_file)
    assert content == ""
