# `garden-ai`

🌱 Hello, Garden 🌱

**Usage**:

```console
$ garden-ai [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--version`
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `whoami`: Print the email of the currently logged in...
* `login`: Attempts to login if the user is currently...
* `logout`: Logs out the current user.
* `mcp`: MCP server commands
* `garden`: Manage Gardens
* `function`: Manage functions (Modal and HPC)

## `garden-ai whoami`

Print the email of the currently logged in user. If logged out, attempt a login.

**Usage**:

```console
$ garden-ai whoami [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `garden-ai login`

Attempts to login if the user is currently logged out.

**Usage**:

```console
$ garden-ai login [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `garden-ai logout`

Logs out the current user.

**Usage**:

```console
$ garden-ai logout [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `garden-ai mcp`

MCP server commands

**Usage**:

```console
$ garden-ai mcp [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `setup`: Add config file for client
* `serve`: Start the Garden MCP server.

### `garden-ai mcp setup`

Add config file for client

**Usage**:

```console
$ garden-ai mcp setup [OPTIONS]
```

**Options**:

* `--client TEXT`: &#x27;claude&#x27;, &#x27;claude code&#x27;, &#x27;gemini&#x27;, &#x27;cursor&#x27;, &#x27;windsurf&#x27;
* `--path TEXT`: Path to initalize config file for any other mcp client
* `--help`: Show this message and exit.

### `garden-ai mcp serve`

Start the Garden MCP server.

**Usage**:

```console
$ garden-ai mcp serve [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `garden-ai garden`

Manage Gardens

**Usage**:

```console
$ garden-ai garden [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `create`: Create a new garden.
* `list`: List gardens.
* `search`: Search for gardens using full-text search.
* `show`: Show details of a specific garden.
* `update`: Update a garden&#x27;s metadata.
* `add-functions`: Add functions to an existing garden.
* `delete`: Delete a garden.

### `garden-ai garden create`

Create a new garden.

**Usage**:

```console
$ garden-ai garden create [OPTIONS]
```

**Options**:

* `-t, --title TEXT`: Title of the garden  [required]
* `-a, --authors TEXT`: Comma-separated list of authors. If not provided, uses the current user.
* `-c, --contributors TEXT`: Comma-separated list of contributors
* `-d, --description TEXT`: Description of the garden
* `--tags TEXT`: Comma-separated list of tags
* `-m, --modal-function-ids TEXT`: Comma-separated Modal function IDs
* `-g, --hpc-function-ids TEXT`: Comma-separated HPC function IDs
* `--year TEXT`: Publication year  [default: 2026]
* `--version TEXT`: Garden version  [default: 0.0.1]
* `--help`: Show this message and exit.

### `garden-ai garden list`

List gardens.

**Usage**:

```console
$ garden-ai garden list [OPTIONS]
```

**Options**:

* `--all`: List all published gardens instead of just yours
* `--tags TEXT`: Filter by comma-separated tags
* `--authors TEXT`: Filter by comma-separated authors
* `--year TEXT`: Filter by year
* `-n, --limit INTEGER`: Maximum results to show  [default: 20]
* `--json`: Output results as JSON
* `--pretty`: Pretty-print JSON output
* `--help`: Show this message and exit.

### `garden-ai garden search`

Search for gardens using full-text search.

**Usage**:

```console
$ garden-ai garden search [OPTIONS] QUERY
```

**Arguments**:

* `QUERY`: Search query  [required]

**Options**:

* `-n, --limit INTEGER`: Maximum results to show  [default: 10]
* `--json`: Output results as JSON
* `--pretty`: Pretty-print JSON output
* `--help`: Show this message and exit.

### `garden-ai garden show`

Show details of a specific garden.

**Usage**:

```console
$ garden-ai garden show [OPTIONS] DOI
```

**Arguments**:

* `DOI`: DOI of the garden to show  [required]

**Options**:

* `--json`: Output results as JSON
* `--pretty`: Pretty-print JSON output
* `--help`: Show this message and exit.

### `garden-ai garden update`

Update a garden&#x27;s metadata.

Note: For list fields (authors, contributors, tags, function IDs), the provided
values will REPLACE the entire existing list. To add or remove individual items,
first retrieve the current values with &#x27;garden show&#x27;, modify as needed, then
provide the complete new list.

**Usage**:

```console
$ garden-ai garden update [OPTIONS] DOI
```

**Arguments**:

* `DOI`: DOI of the garden to update  [required]

**Options**:

* `-t, --title TEXT`: New title
* `-a, --authors TEXT`: Comma-separated list of authors. Replaces the entire authors list.
* `-c, --contributors TEXT`: Comma-separated list of contributors. Replaces the entire contributors list.
* `-d, --description TEXT`: New description
* `--tags TEXT`: Comma-separated list of tags. Replaces the entire tags list.
* `--version TEXT`: New version
* `-m, --modal-function-ids TEXT`: Comma-separated Modal function IDs. Replaces the entire list.
* `-g, --hpc-function-ids TEXT`: Comma-separated HPC function IDs. Replaces the entire list.
* `--help`: Show this message and exit.

### `garden-ai garden add-functions`

Add functions to an existing garden.

**Usage**:

```console
$ garden-ai garden add-functions [OPTIONS] DOI
```

**Arguments**:

* `DOI`: DOI of the garden  [required]

**Options**:

* `-m, --modal-function-ids TEXT`: Comma-separated Modal function IDs to add
* `-g, --hpc-function-ids TEXT`: Comma-separated HPC function IDs to add
* `--replace`: Replace existing functions instead of adding
* `--help`: Show this message and exit.

### `garden-ai garden delete`

Delete a garden.

**Usage**:

```console
$ garden-ai garden delete [OPTIONS] DOI
```

**Arguments**:

* `DOI`: DOI of the garden to delete  [required]

**Options**:

* `-f, --force`: Skip confirmation prompt
* `--help`: Show this message and exit.

## `garden-ai function`

Manage functions (Modal and HPC)

**Usage**:

```console
$ garden-ai function [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `modal`: Manage Modal functions and apps
* `hpc`: Manage HPC functions

### `garden-ai function modal`

Manage Modal functions and apps

**Usage**:

```console
$ garden-ai function modal [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `list`: List your Modal functions.
* `show`: Show details of a Modal function.
* `update`: Update a Modal function&#x27;s metadata.
* `app`: Manage Modal apps

#### `garden-ai function modal list`

List your Modal functions.

**Usage**:

```console
$ garden-ai function modal list [OPTIONS]
```

**Options**:

* `-n, --limit INTEGER`: Maximum results  [default: 50]
* `--json`: Output results as JSON
* `--pretty`: Pretty-print JSON output
* `--help`: Show this message and exit.

#### `garden-ai function modal show`

Show details of a Modal function.

**Usage**:

```console
$ garden-ai function modal show [OPTIONS] FUNCTION_ID
```

**Arguments**:

* `FUNCTION_ID`: Modal function ID  [required]

**Options**:

* `-c, --code`: Show function code
* `--json`: Output results as JSON
* `--pretty`: Pretty-print JSON output
* `--help`: Show this message and exit.

#### `garden-ai function modal update`

Update a Modal function&#x27;s metadata.

**Usage**:

```console
$ garden-ai function modal update [OPTIONS] FUNCTION_ID
```

**Arguments**:

* `FUNCTION_ID`: Modal function ID  [required]

**Options**:

* `-t, --title TEXT`: New title
* `-d, --description TEXT`: New description
* `-a, --authors TEXT`: New comma-separated authors
* `--tags TEXT`: New comma-separated tags
* `--help`: Show this message and exit.

#### `garden-ai function modal app`

Manage Modal apps

**Usage**:

```console
$ garden-ai function modal app [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `deploy`: Deploy a Modal app from a Python file.
* `list`: List your Modal apps.
* `show`: Show details of a Modal app.
* `delete`: Delete a Modal app and its functions.

##### `garden-ai function modal app deploy`

Deploy a Modal app from a Python file.

**Usage**:

```console
$ garden-ai function modal app deploy [OPTIONS] FILE
```

**Arguments**:

* `FILE`: Path to Modal Python file  [required]

**Options**:

* `-n, --name TEXT`: App name (auto-detected if not provided)
* `-t, --title TEXT`: Title for functions (defaults to app name)
* `-a, --authors TEXT`: Comma-separated list of authors
* `--tags TEXT`: Comma-separated list of tags
* `--base-image TEXT`: Base Docker image  [default: python:3.11-slim]
* `-r, --requirements TEXT`: Comma-separated pip requirements
* `--wait / --no-wait`: Wait for deployment to complete  [default: wait]
* `--timeout FLOAT`: Deployment timeout in seconds  [default: 300.0]
* `--help`: Show this message and exit.

##### `garden-ai function modal app list`

List your Modal apps.

**Usage**:

```console
$ garden-ai function modal app list [OPTIONS]
```

**Options**:

* `--json`: Output results as JSON
* `--pretty`: Pretty-print JSON output
* `--help`: Show this message and exit.

##### `garden-ai function modal app show`

Show details of a Modal app.

**Usage**:

```console
$ garden-ai function modal app show [OPTIONS] APP_ID
```

**Arguments**:

* `APP_ID`: Modal app ID  [required]

**Options**:

* `-c, --code`: Show file contents
* `--show-app-text`: Show the app_text field (deployed code)
* `--json`: Output results as JSON
* `--pretty`: Pretty-print JSON output
* `--help`: Show this message and exit.

##### `garden-ai function modal app delete`

Delete a Modal app and its functions.

**Usage**:

```console
$ garden-ai function modal app delete [OPTIONS] APP_ID
```

**Arguments**:

* `APP_ID`: Modal app ID to delete  [required]

**Options**:

* `-f, --force`: Skip confirmation
* `--help`: Show this message and exit.

### `garden-ai function hpc`

Manage HPC functions

**Usage**:

```console
$ garden-ai function hpc [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `deploy`: Deploy a HPC function from a Python file.
* `list`: List your HPC functions.
* `show`: Show details of a HPC function.
* `update`: Update a HPC function&#x27;s metadata.
* `delete`: Delete a HPC function.
* `endpoint`: Manage HPC endpoints

#### `garden-ai function hpc deploy`

Deploy a HPC function from a Python file.

**Usage**:

```console
$ garden-ai function hpc deploy [OPTIONS] FILE
```

**Arguments**:

* `FILE`: Path to Python file with function  [required]

**Options**:

* `-n, --name TEXT`: Function name (auto-detected if not provided)
* `-t, --title TEXT`: Function title
* `-e, --endpoint-ids TEXT`: Comma-separated endpoint IDs  [required]
* `-a, --authors TEXT`: Comma-separated authors
* `-d, --description TEXT`: Function description
* `--tags TEXT`: Comma-separated tags
* `-r, --requirements TEXT`: Comma-separated pip requirements
* `--help`: Show this message and exit.

#### `garden-ai function hpc list`

List your HPC functions.

**Usage**:

```console
$ garden-ai function hpc list [OPTIONS]
```

**Options**:

* `--json`: Output results as JSON
* `--pretty`: Pretty-print JSON output
* `--help`: Show this message and exit.

#### `garden-ai function hpc show`

Show details of a HPC function.

**Usage**:

```console
$ garden-ai function hpc show [OPTIONS] FUNCTION_ID
```

**Arguments**:

* `FUNCTION_ID`: Function ID  [required]

**Options**:

* `-c, --code`: Show function code
* `--json`: Output results as JSON
* `--pretty`: Pretty-print JSON output
* `--help`: Show this message and exit.

#### `garden-ai function hpc update`

Update a HPC function&#x27;s metadata.

**Usage**:

```console
$ garden-ai function hpc update [OPTIONS] FUNCTION_ID
```

**Arguments**:

* `FUNCTION_ID`: Function ID  [required]

**Options**:

* `-n, --name TEXT`: New function name
* `-t, --title TEXT`: New title
* `-d, --description TEXT`: New description
* `-a, --authors TEXT`: New comma-separated authors
* `--tags TEXT`: New comma-separated tags
* `-e, --endpoint-ids TEXT`: New comma-separated endpoint IDs
* `--help`: Show this message and exit.

#### `garden-ai function hpc delete`

Delete a HPC function.

**Usage**:

```console
$ garden-ai function hpc delete [OPTIONS] FUNCTION_ID
```

**Arguments**:

* `FUNCTION_ID`: Function ID  [required]

**Options**:

* `-f, --force`: Skip confirmation
* `--help`: Show this message and exit.

#### `garden-ai function hpc endpoint`

Manage HPC endpoints

**Usage**:

```console
$ garden-ai function hpc endpoint [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `create`: Register a new HPC endpoint.
* `list`: List available HPC endpoints.
* `show`: Show details of a HPC endpoint.
* `update`: Update a HPC endpoint.
* `delete`: Delete a HPC endpoint.

##### `garden-ai function hpc endpoint create`

Register a new HPC endpoint.

**Usage**:

```console
$ garden-ai function hpc endpoint create [OPTIONS]
```

**Options**:

* `-n, --name TEXT`: Endpoint name  [required]
* `-g, --gcmu-id TEXT`: Globus Compute endpoint UUID
* `--help`: Show this message and exit.

##### `garden-ai function hpc endpoint list`

List available HPC endpoints.

**Usage**:

```console
$ garden-ai function hpc endpoint list [OPTIONS]
```

**Options**:

* `-n, --limit INTEGER`: Maximum results  [default: 50]
* `--json`: Output results as JSON
* `--pretty`: Pretty-print JSON output
* `--help`: Show this message and exit.

##### `garden-ai function hpc endpoint show`

Show details of a HPC endpoint.

**Usage**:

```console
$ garden-ai function hpc endpoint show [OPTIONS] ENDPOINT_ID
```

**Arguments**:

* `ENDPOINT_ID`: Endpoint ID  [required]

**Options**:

* `--json`: Output results as JSON
* `--pretty`: Pretty-print JSON output
* `--help`: Show this message and exit.

##### `garden-ai function hpc endpoint update`

Update a HPC endpoint.

**Usage**:

```console
$ garden-ai function hpc endpoint update [OPTIONS] ENDPOINT_ID
```

**Arguments**:

* `ENDPOINT_ID`: Endpoint ID  [required]

**Options**:

* `-n, --name TEXT`: New name
* `-g, --gcmu-id TEXT`: New GCMU ID
* `--help`: Show this message and exit.

##### `garden-ai function hpc endpoint delete`

Delete a HPC endpoint.

**Usage**:

```console
$ garden-ai function hpc endpoint delete [OPTIONS] ENDPOINT_ID
```

**Arguments**:

* `ENDPOINT_ID`: Endpoint ID  [required]

**Options**:

* `-f, --force`: Skip confirmation
* `--help`: Show this message and exit.
