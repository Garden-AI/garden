import json
import os
import platform
import shutil
import subprocess
from json import JSONDecodeError
from pathlib import Path

try:
    # Catch ImportError when mcp extra is not installed
    from uv import find_uv_bin
except ImportError:
    pass


class MCPConfigInitalizer:
    @staticmethod
    def setup_claude():
        system = platform.system()
        if system == "Darwin":  # macOS
            config_path = (
                Path.home()
                / "Library"
                / "Application Support"
                / "Claude"
                / "claude_desktop_config.json"
            )
        elif system == "Windows":
            config_path = (
                Path(os.environ.get("APPDATA", ""))
                / "Claude"
                / "claude_desktop_config.json"
            )
        else:  # Linux
            config_path = (
                Path.home() / ".config" / "Claude" / "claude_desktop_config.json"
            )

        return MCPConfigInitalizer._initialize_config_file(config_path)

    @staticmethod
    def setup_claude_code():
        claude_path = shutil.which("claude")  # Find claude executable on current path
        if not claude_path:
            raise FileNotFoundError("claude-code executable not found on current path")

        claude_exe = Path(claude_path)
        uv_exe = find_uv_bin()

        command = [
            str(claude_exe),
            "mcp",
            "add",
            "garden",
            "--",
            uv_exe,
            "run",
            "--with",
            "garden-ai[mcp]",
            "garden-ai",
            "mcp",
            "serve",
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        if result.stdout:
            # Parse claude response for file path
            parts = result.stdout.split("File modified: ")
            if len(parts) > 1:
                # Get everything after "File modified: " and split on whitespace
                path = parts[1].split()[0]

            return path
        elif result.stderr:
            raise ValueError(result.stderr)

    @staticmethod
    def setup_cursor():
        # Cursor config path
        config_path = Path.home() / ".cursor" / "mcp.json"

        return MCPConfigInitalizer._initialize_config_file(config_path)

    @staticmethod
    def setup_gemini():
        # Gemini CLI config path
        config_path = Path.home() / ".gemini" / "settings.json"

        return MCPConfigInitalizer._initialize_config_file(config_path)

    @staticmethod
    def setup_windsurf():
        # Windsurf config path
        config_path = Path.home() / ".codeium" / "windsurf" / "mcp_config.json"

        return MCPConfigInitalizer._initialize_config_file(config_path)

    @staticmethod
    def setup_custom(path: str):
        return MCPConfigInitalizer._initialize_config_file(Path(path))

    @staticmethod
    def _initialize_config_file(config_path: Path):
        resolved_config_path = config_path.resolve()

        if resolved_config_path.exists():
            with open(resolved_config_path, "r") as f:
                try:
                    config = json.load(f)
                except JSONDecodeError as e:
                    raise JSONDecodeError(
                        f"Invalid existing json: {e.msg}", e.doc, e.pos
                    )

        else:
            raise FileNotFoundError(
                f"Configuration file does not exist at: {resolved_config_path}"
            )

        if "mcpServers" not in config:
            config["mcpServers"] = {}

        uv_exe = find_uv_bin()

        config["mcpServers"]["garden-ai"] = {
            "command": uv_exe,
            "args": ["run", "--with", "garden-ai[mcp]", "garden-ai", "mcp", "serve"],
        }

        with open(resolved_config_path, "w") as f:
            json.dump(config, f, indent=2)

        return str(resolved_config_path)
