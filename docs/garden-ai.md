# `garden-ai`

The garden-ai package is mostly used programatically from Python. But there are some helper CLI commands that let you examine Garden's login state on your machine.

**Usage**:

```console
$ garden-ai [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--version`
* `--help`: Show this message and exit.

**Commands**:

* `login`: Attempts to login if the user is currently...
* `logout`: Logs out the current user.
* `whoami`: Print the email of the currently logged in...

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


## `garden-ai whoami`

Print the email of the currently logged in user. If logged out, attempt a login.

**Usage**:

```console
$ garden-ai whoami [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.
