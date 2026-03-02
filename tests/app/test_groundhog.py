"""Tests for groundhog CLI parsing functions."""

import tempfile
from pathlib import Path

from garden_ai.app.groundhog import _extract_function_from_file

# Example script with @hog.function
SCRIPT_WITH_FUNCTION = '''
import groundhog_hpc as hog

@hog.function(endpoint="anvil")
def compute_mean(numbers):
    """Calculate the mean of numbers."""
    import numpy as np
    return float(np.mean(numbers))

@hog.harness()
def main():
    result = compute_mean.remote([1, 2, 3])
    print(result)
'''

# Example script with @hog.method in a class
SCRIPT_WITH_METHOD = '''
import groundhog_hpc as hog

class Statistics:
    @hog.method(endpoint="anvil")
    def compute_mean(numbers):
        """Calculate mean using numpy."""
        import numpy as np
        return float(np.mean(numbers))

    @hog.method(endpoint="anvil")
    def compute_std(numbers):
        """Calculate standard deviation."""
        import numpy as np
        return float(np.std(numbers))
'''

# Script with no hog decorators
SCRIPT_NO_DECORATORS = '''
def helper_function():
    """Just a helper."""
    pass
'''


def _write_temp_script(content: str, name: str = "script") -> Path:
    """Write content to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix=f"{name}_"
    ) as f:
        f.write(content)
        return Path(f.name)


def test_extracts_hog_function():
    """@hog.function decorated functions are extracted with name and docstring."""
    path = _write_temp_script(SCRIPT_WITH_FUNCTION)
    result = _extract_function_from_file(path)

    assert result["function_name"] == "compute_mean"
    assert result["docstring"] == "Calculate the mean of numbers."
    assert "import groundhog_hpc" in result["function_text"]


def test_extracts_hog_method_with_class_name():
    """@hog.method decorated methods include ClassName.method_name format."""
    path = _write_temp_script(SCRIPT_WITH_METHOD)
    result = _extract_function_from_file(path)

    assert result["function_name"] == "Statistics.compute_mean"
    assert result["docstring"] == "Calculate mean using numpy."


def test_ignores_harness_decorator():
    """@hog.harness is for orchestration, not remote execution - should be skipped."""
    path = _write_temp_script(SCRIPT_WITH_FUNCTION)
    result = _extract_function_from_file(path)

    # Should find compute_mean, not main (the harness)
    assert result["function_name"] == "compute_mean"


def test_falls_back_to_filename():
    """Files without hog decorators fall back to filename."""
    path = _write_temp_script(SCRIPT_NO_DECORATORS, name="my_model")
    result = _extract_function_from_file(path)

    assert result["function_name"] == path.stem
    assert result["docstring"] == ""
