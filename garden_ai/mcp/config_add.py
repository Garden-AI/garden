import os
import json
from pathlib import Path
import subprocess
import platform


class MCPConfigInitalizer:
    @staticmethod
    def claude():
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
    def claude_code():
        command = "claude mcp add garden -- uv run --with garden-ai[mcp] garden-ai mcp"

        result = subprocess.run(command.split(" "), capture_output=True, text=True)

        if result.stdout:
            # Parse claude response for file path
            parts = result.stdout.split("File modified: ")
            if len(parts) > 1:
                # Get everything after "File modified: " and split on whitespace
                path = parts[1].split()[0]

            return path
        else:
            raise ValueError("'garden' mcp config already exists")

    @staticmethod
    def cursor():
        # Cursor config path
        config_path = Path.home() / ".cursor" / "mcp.json"

        return MCPConfigInitalizer._initialize_config_file(config_path)

    @staticmethod
    def gemini():
        # Gemini CLI config path
        config_path = Path.home() / ".gemini" / "settings.json"

        return MCPConfigInitalizer._initialize_config_file(config_path)

    @staticmethod
    def windsurf():
        # Windsurf config path
        config_path = Path.home() / ".codeium" / "windsurf" / "mcp_config.json"

        return MCPConfigInitalizer._initialize_config_file(config_path)

    @staticmethod
    def custom(path: str):
        return MCPConfigInitalizer._initialize_config_file(Path(path))

    @staticmethod
    def _initialize_config_file(config_path: Path):
        config_path.parent.mkdir(parents=True, exist_ok=True)

        if config_path.exists():
            with open(config_path, "r") as f:
                # If file is empty or contains invalid JSON, start with empty config
                try:
                    config = json.load(f)
                except json.JSONDecodeError:
                    config = {}
        else:
            config = {}

        if "mcpServers" not in config:
            config["mcpServers"] = {}

        config["mcpServers"]["garden-ai"] = {
            "command": "uv",
            "args": ["run", "--with", "garden-ai[mcp]", "garden-ai", "mcp"],
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        return str(config_path)
