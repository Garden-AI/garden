# Garden MCP server

This is a prototype implementation of a local (stdio) MCP server to enable MCP
clients like claude desktop to interact with the garden service via the sdk
(instead of via a hosted MCP server)

It exposes the following tools (TODO):

- [ ] search_gardens
- [ ] search_functions
- [ ] call_function

# Prerequisites

- uv
- `uvx garden-ai login` for auth

## claude desktop config

```
{
  "mcpServers": {
    "garden": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "garden[mcp]",
        "garden-ai",
        "mcp"
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
        "--with",
        "/path/to/local/repo/garden[mcp]",
        "garden-ai",
        "mcp"
      ]
    }
  }
}
```
