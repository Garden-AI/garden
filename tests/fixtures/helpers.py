from pathlib import Path


def get_fixture_file_path(filename):
    current_file = Path(__file__)
    fixtures_directory = current_file.parent
    fixture_file_path = fixtures_directory / filename
    return fixture_file_path
