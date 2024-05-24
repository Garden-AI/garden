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

* `docker`
* `entrypoint`: sub-commands for creating and manipulating...
* `garden`: sub-commands for creating and manipulating...
* `login`: Attempts to login if the user is currently...
* `logout`: Logs out the current user.
* `notebook`: sub-commands for editing and publishing...
* `whoami`: Print the email of the currently logged in...

## `garden-ai docker`

**Usage**:

```console
$ garden-ai docker [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `check`: Check if Garden can access Docker on your...
* `prune`: Remove Garden-related Docker images,...

### `garden-ai docker check`

Check if Garden can access Docker on your computer.

If Garden can't access Docker and it's not clear what the problem is,
using --verbose will print out a full stack trace.

**Usage**:

```console
$ garden-ai docker check [OPTIONS]
```

**Options**:

* `-v, --verbose`
* `--help`: Show this message and exit.

### `garden-ai docker prune`

Remove Garden-related Docker images, freeing up disk space.

**Usage**:

```console
$ garden-ai docker prune [OPTIONS]
```

**Options**:

* `--keep-base`: If enabled, keep official gardenai/base images and only remove custom user images (e.g. those created by publishing)
* `--dry-run`: If enabled, just print the tags of images that would be pruned.
* `--remove-dangling`: Remove any dangling images. This includes dangling images that may not be Garden-related.
* `--help`: Show this message and exit.

## `garden-ai entrypoint`

sub-commands for creating and manipulating entrypoints

**Usage**:

```console
$ garden-ai entrypoint [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `add-paper`
* `add-repository`
* `edit`: Edit an Entrypoint's metadata
* `list`: Lists all local entrypoints.
* `register-doi`: Moves an Entrypoint's DOI out of draft state.
* `show`: Shows all info for some entrypoints

### `garden-ai entrypoint add-paper`

**Usage**:

```console
$ garden-ai entrypoint add-paper [OPTIONS]
```

**Options**:

* `-d, --doi TEXT`: The DOI for the entrypoint you would like to link a paper to  [required]
* `-t, --title TEXT`: [required]
* `-a, --author TEXT`: Acknowledge an author in this repository. Repeat to indicate multiple (like --author).
* `-p, --paper-doi TEXT`: Optional, the digital identifier that the paper may be linked to
* `-c, --citation TEXT`: Optional, enter how the paper may be cited.
* `--help`: Show this message and exit.

### `garden-ai entrypoint add-repository`

**Usage**:

```console
$ garden-ai entrypoint add-repository [OPTIONS]
```

**Options**:

* `-d, --doi TEXT`: The DOI for the entrypoint you would like to add a repository to  [required]
* `-u, --url TEXT`: [required]
* `-r, --repository_name TEXT`: [required]
* `-c, --contributor TEXT`: Acknowledge a contributor in this repository. Repeat to indicate multiple (like --author).
* `--help`: Show this message and exit.

### `garden-ai entrypoint edit`

Edit an Entrypoint's metadata

**Usage**:

```console
$ garden-ai entrypoint edit [OPTIONS] DOI
```

**Arguments**:

* `DOI`: The DOI of the entrypoint you want to edit  [required]

**Options**:

* `--help`: Show this message and exit.

### `garden-ai entrypoint list`

Lists all local entrypoints.

**Usage**:

```console
$ garden-ai entrypoint list [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `garden-ai entrypoint register-doi`

Moves an Entrypoint's DOI out of draft state.

Parameters
----------
doi : str
    The DOI of the entrypoint to be registered.

**Usage**:

```console
$ garden-ai entrypoint register-doi [OPTIONS] DOI
```

**Arguments**:

* `DOI`: The draft entrypoint DOI you want to register  [required]

**Options**:

* `--help`: Show this message and exit.

### `garden-ai entrypoint show`

Shows all info for some entrypoints

**Usage**:

```console
$ garden-ai entrypoint show [OPTIONS] ENTRYPOINT_IDS...
```

**Arguments**:

* `ENTRYPOINT_IDS...`: The DOIs of the entrypoints you want to show local data for.   [required]

**Options**:

* `--help`: Show this message and exit.

## `garden-ai garden`

sub-commands for creating and manipulating Gardens

**Usage**:

```console
$ garden-ai garden [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `add-entrypoint`: Add a registered entrypoint to a garden
* `create`: Create a new Garden
* `delete`: Delete a Garden from your local storage...
* `edit`: Edit a Garden's metadata
* `list`: Lists all local Gardens.
* `publish`: Push data about a Garden stored to Globus...
* `register-doi`: Moves a Garden's DOI out of draft state.
* `search`: Queries the Garden search index and prints...
* `show`: Shows all info for some Gardens

### `garden-ai garden add-entrypoint`

Add a registered entrypoint to a garden

**Usage**:

```console
$ garden-ai garden add-entrypoint [OPTIONS]
```

**Options**:

* `-g, --garden TEXT`: The name of the garden you want to add an entrypoint to  [required]
* `-p, --entrypoint TEXT`: The name of the entrypoint you want to add  [required]
* `-a, --alias TEXT`: Alternate short_name to use when calling this entrypoint as a "method" of thegarden, e.g. ``my_garden.alias(args, endpoint=...)``. Defaults to the variablename used when the entrypoint was first registered.
* `--help`: Show this message and exit.

### `garden-ai garden create`

Create a new Garden

**Usage**:

```console
$ garden-ai garden create [OPTIONS]
```

**Options**:

* `-t, --title TEXT`: Provide an official title (as it should appear in citations)  [required]
* `-a, --author TEXT`: Name an author of this Garden. Repeat this to indicate multiple authors: `garden create ... --author='Mendel, Gregor' --author 'Other-Author, Anne' ...` (order is preserved).
* `-y, --year TEXT`: [default: 2024]
* `-c, --contributor TEXT`: Acknowledge a contributor in this Garden. Repeat to indicate multiple (like --author).
* `-d, --description TEXT`: A brief summary of the Garden and/or its purpose, to aid discovery by other Gardeners.
* `--tag TEXT`: Add a tag, keyword, key phrase or other classification pertaining to the Garden.
* `--verbose / --no-verbose`: If true, pretty-print Garden's metadata when created.  [default: no-verbose]
* `--help`: Show this message and exit.

### `garden-ai garden delete`

Delete a Garden from your local storage and the thegardens.ai website

**Usage**:

```console
$ garden-ai garden delete [OPTIONS]
```

**Options**:

* `-g, --garden TEXT`: The DOI of the garden you want to publish  [required]
* `--help`: Show this message and exit.

### `garden-ai garden edit`

Edit a Garden's metadata

**Usage**:

```console
$ garden-ai garden edit [OPTIONS] DOI
```

**Arguments**:

* `DOI`: The DOI of the garden you want to edit  [required]

**Options**:

* `--help`: Show this message and exit.

### `garden-ai garden list`

Lists all local Gardens.

**Usage**:

```console
$ garden-ai garden list [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `garden-ai garden publish`

Push data about a Garden stored to Globus Search so that other users can search for it

**Usage**:

```console
$ garden-ai garden publish [OPTIONS]
```

**Options**:

* `-g, --garden TEXT`: The DOI of the garden you want to publish  [required]
* `--help`: Show this message and exit.

### `garden-ai garden register-doi`

Moves a Garden's DOI out of draft state.

Parameters
----------
doi : str
    The DOI of the garden to be registered.

**Usage**:

```console
$ garden-ai garden register-doi [OPTIONS] DOI
```

**Arguments**:

* `DOI`: The draft garden DOI you want to register  [required]

**Options**:

* `--help`: Show this message and exit.

### `garden-ai garden search`

Queries the Garden search index and prints matching results. All query components are ANDed together.
So if you say `garden-ai garden search --description "foo" --title "bar"` you will get results
for gardens that have "foo" in their description and "bar" in their title.

**Usage**:

```console
$ garden-ai garden search [OPTIONS]
```

**Options**:

* `-t, --title TEXT`: Title of a Garden
* `-a, --author TEXT`: an author of the Garden
* `-y, --year TEXT`: year the Garden was published
* `-c, --contributor TEXT`: a contributor to the Garden
* `-d, --description TEXT`: text in the description of the Garden you are searching for
* `--tag TEXT`: A tag of the Garden
* `--verbose / --no-verbose`: If true, print the query being passed to Globus Search.  [default: no-verbose]
* `--raw-query TEXT`: Form your own Globus Search query directly. It will be passed to Search in advanced mode.Overrides all the other query options.See https://docs.globus.org/api/search/reference/get_query for more details.
* `--help`: Show this message and exit.

### `garden-ai garden show`

Shows all info for some Gardens

**Usage**:

```console
$ garden-ai garden show [OPTIONS] GARDEN_IDS...
```

**Arguments**:

* `GARDEN_IDS...`: The DOIs of the Gardens you want to show the local data for. e.g. ``garden show garden1_doi garden2_doi`` will show the local data for both Gardens listed.  [required]

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

## `garden-ai notebook`

sub-commands for editing and publishing from sandboxed notebooks.

**Usage**:

```console
$ garden-ai notebook [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `debug`: Open the debugging notebook in a...
* `list-premade-images`: List all Garden base docker images
* `publish`
* `start`: Open a notebook file in a sandboxed...

### `garden-ai notebook debug`

Open the debugging notebook in a pre-prepared container.

Changes to the notebook file will NOT persist after the container shuts down.
Quit the process with Ctrl-C or by shutting down jupyter from the browser.

**Usage**:

```console
$ garden-ai notebook debug [OPTIONS] PATH
```

**Arguments**:

* `PATH`: Path to a .ipynb notebook whose remote environment will be approximated for debugging.  [required]

**Options**:

* `--requirements PATH`: Path to a requirements.txt containing additional dependencies to install in the base image.
* `--help`: Show this message and exit.

### `garden-ai notebook list-premade-images`

List all Garden base docker images

**Usage**:

```console
$ garden-ai notebook list-premade-images [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `garden-ai notebook publish`

**Usage**:

```console
$ garden-ai notebook publish [OPTIONS] PATH
```

**Arguments**:

* `PATH`: [required]

**Options**:

* `--requirements PATH`: Path to a requirements.txt containing additional dependencies to install in the base image.
* `--base-image TEXT`: A Garden base image to run your notebook inside of. This will be the foundation for the image that runs your entrypoints.For example, to run on top of the default Garden python 3.8 image, use --base-image 3.8-base. To see all the available Garden base images, use 'garden-ai notebook list-premade-images'
* `--doi TEXT`: A DOI of a Garden that EACH entrypoint in the notebook should be added too. This is considered to be a global notebook DOI. To override the global notebook DOI for a specific entrypoint, provide the garden_entrypoint decorator with the optional garden_doi argument.
* `-v, --verbose`
* `--keep-outputs`: By default, Garden will clear all cell outputs before publishing. If you would like to have your cell outputs visible on the UI, use this flag.
* `--help`: Show this message and exit.

### `garden-ai notebook start`

Open a notebook file in a sandboxed environment. Optionally, specify a different base docker image.

Changes to the notebook file will persist after the container shuts down.
Quit the process with Ctrl-C or by shutting down jupyter from the browser.
If a different base image is chosen, that image will be reused as the default for this notebook in the future.

**Usage**:

```console
$ garden-ai notebook start [OPTIONS] [PATH]
```

**Arguments**:

* `[PATH]`: Path to a .ipynb notebook to open in a fresh, isolated container.

**Options**:

* `--base-image TEXT`: A Garden base image to boot the notebook in. For example, to boot your notebook with the default Garden python 3.8 image, use --base-image 3.8-base. To see all the available Garden base images, use 'garden-ai notebook list-premade-images'
* `--requirements PATH`: Path to a requirements.txt containing additional dependencies to install in the base image.
* `--doi TEXT`: A DOI of a Garden that EACH entrypoint in the notebook should be added too. This is considered to be a global notebook DOI. To override the global notebook DOI for a specific entrypoint, provide the garden_entrypoint decorator with the optional garden_doi argument.
* `--help`: Show this message and exit.

## `garden-ai whoami`

Print the email of the currently logged in user. If logged out, attempt a login.

**Usage**:

```console
$ garden-ai whoami [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.
