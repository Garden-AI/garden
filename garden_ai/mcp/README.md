# Garden MCP server

This is a prototype implementation of a local (stdio) MCP server to enable MCP clients like claude desktop to interact with the garden service via the sdk (instead of via a hosted MCP server)

It exposes the following tools (TODO):

- [ ] search_gardens
- [ ] search_functions
- [ ] call_function
- [ ] MLIP-demo-related tools (see below)

## Prerequisites

- uv
- `uvx garden-ai login` to set up auth

## Quick Setup

Use the automated setup command to configure your MCP client:

```bash
# For Claude Desktop
garden-ai mcp setup --client claude

# For Claude Code
garden-ai mcp setup --client claude-code  

# For other clients (cursor, windsurf, gemini)
garden-ai mcp setup --client <client-name>

# For custom config file
garden-ai mcp setup --path /path/to/your/config.json
```

This command automatically adds the Garden MCP server configuration to your client's config file.

## claude desktop config

```
{
  "mcpServers": {
    "garden": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "garden-ai[mcp]",
        "garden-ai",
        "mcp",
        "serve"
      ]
    }
  }
}
```

### including MLIP tools

To include the bespoke `submit_relaxation_job`, `check_job_status` and `get_job_results` tools, add an additional `--with garden-ai[mlip]` to the uv command, like so:

```
{
  "mcpServers": {
    "garden": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "garden-ai[mcp]",
        "--with",
        "garden-ai[mlip]",
        "garden-ai",
        "mcp",
        "serve"
      ]
    }
  }
}

```

## local dev claude config

```
{
  "mcpServers": {
    "garden": {
      "command": "uv",
      "args": [
        "run",
        "--no-cache",
        "--with",
        "/path/to/local/repo/garden[mcp]",
        "garden-ai",
        "mcp",
        "serve"
      ],
      "env": {
        "GARDEN_ENV":"dev"
      }
    }
  }
}
```
