import math
from pathlib import Path
import random

import pytest

from garden_ai.hpc_gardens.utils import generate_xyz_str_chunks


@pytest.fixture
def large_xyz_file(tmpdir):
    symbols = ["Cu", "Si", "Pb"]
    nstructures = 10000
    natoms = 100

    tmpfile = tmpdir.join("test.xyz")
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


def test_xyz_chunks_are_under_size_limit(large_xyz_file):
    limit = 5 * 1000 * 1000  # 5MB
    num_chunks = 0
    expected_num_chunks = math.ceil(Path(large_xyz_file).stat().st_size / limit)
    for chunk in generate_xyz_str_chunks(large_xyz_file, max_size_bytes=limit):
        num_chunks += 1
        assert len(chunk.encode("utf-8")) <= limit
    assert num_chunks == expected_num_chunks


def test_empty_file_returns_zero_chunks(tmpdir):
    tmpfile = tmpdir.join("empty.xyz")
    Path(tmpfile).touch()  # make sure the file exists
    expected_num_chunks = 0
    num_chunks = 0
    for _ in generate_xyz_str_chunks(tmpfile):
        num_chunks += 1
    assert expected_num_chunks == num_chunks


def test_xyz_chunk_generator_raises_if_file_does_not_exist(tmpdir):
    tmpfile = tmpdir.join("xyz")
    assert not Path(tmpfile).exists()

    with pytest.raises(FileNotFoundError):
        for chunk in generate_xyz_str_chunks(tmpfile):
            print(chunk)
