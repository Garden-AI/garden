import inspect
from pathlib import Path
from typing import Callable


class BenchmarkScriptBuilder:
    """Helper to build a self-contained benchmark script from a template."""

    def __init__(self, template_path: str | Path = None):
        if template_path is None:
            # Default to the base_runner.py in templates
            template_path = (
                Path(__file__).parent.parent / "templates" / "base_runner.py"
            )

        self.template_path = Path(template_path)
        self.imports = set()
        self.functions = []
        self.preamble = []

    def add_import(self, import_stmt: str):
        """Add an import statement (e.g. 'import numpy as np')."""
        self.imports.add(import_stmt)
        return self

    def add_preamble(self, code: str):
        """Add arbitrary code to the top of the script (after imports)."""
        self.preamble.append(code)
        return self

    def add_function(self, func: Callable, name: str = None):
        """Add a function definition to the script.

        The function source code is inspected and appended.
        If name is provided, the function definition is renamed.
        """
        source = inspect.getsource(func)

        if name:
            import re

            # Replace 'def old_name(' with 'def new_name('
            # This is a simple regex replacement, assuming standard formatting
            pattern = r"def\s+" + func.__name__ + r"\s*\("
            replacement = f"def {name}("
            source = re.sub(pattern, replacement, source, count=1)

        self.functions.append(source)
        return self

    def build(self) -> str:
        """Assemble the final script."""
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found at {self.template_path}")

        template_content = self.template_path.read_text()

        # Construct sections
        imports_block = "\n".join(sorted(self.imports))
        preamble_block = "\n".join(self.preamble)
        functions_block = "\n\n".join(self.functions)

        # We inject our custom code BEFORE the template's main execution logic
        # but AFTER the template's own imports (which are inside the file).
        # Actually, the template has imports at the top. We should probably prepend ours.

        # Simple strategy: Prepend everything to the template, but the template
        # has "USER DEFINED FUNCTIONS" placeholders. We can just append our functions
        # before the main block?

        # Better strategy: The template is designed to have functions injected.
        # Let's just put imports at the top, then functions, then the template content.
        # But we need to be careful about imports in the template.

        final_script = f"""
# ------------------------------------------------------------------------------
# INJECTED IMPORTS
# ------------------------------------------------------------------------------
{imports_block}

# ------------------------------------------------------------------------------
# INJECTED PREAMBLE
# ------------------------------------------------------------------------------
{preamble_block}

# ------------------------------------------------------------------------------
# INJECTED FUNCTIONS
# ------------------------------------------------------------------------------
{functions_block}

# ------------------------------------------------------------------------------
# BASE RUNNER TEMPLATE
# ------------------------------------------------------------------------------
{template_content}
"""
        return final_script
