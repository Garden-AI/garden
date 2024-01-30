# Installing `garden-ai`

The Garden CLI (`garden-ai`) is a Python package that's published on PyPI. You can install it however you normally install Python packages.

## Recommended Method: pipx

pipx is the easiest and best way to install Garden. pipx is a tool for easily installing and running Python command line applications in an isolated environment. You can think of it like `venv` or `conda` for the special case of managing CLI apps. [You can learn how to install pipx on its GitHub page.](https://github.com/pypa/pipx?tab=readme-ov-file#pipx--install-and-run-python-applications-in-isolated-environments)

Once you have pipx installed, install Garden.

```
pipx install garden-ai
```

You can keep up with new releases too.

```
pipx upgrade garden-ai
```

## Alternate Method: Set Up a Virtual Environment

To install with venv:

```bash
mkdir my_garden_project
cd my_garden_project
python3 -m venv garden-env
source myenv/bin/activate
pip install garden-ai
```

## Verify Your Installation

To verify your installation, you can check the version of Garden:

```bash
garden-ai --version
```

You should see the version of your installed Garden package printed to the console.

## Optional: Tab Completion

The Garden CLI also supports tab completion to show options of local entrypoints and gardens.

If you want to use Garden with tab completion, run:
```bash
garden-ai --install-completion
```
> [!NOTE] Note
> ``--install-completion`` should already modify your ``.zshrc``, but in some cases the following additional configuration has been necessary for ``zsh`` users. Ensure it matches exactly to:
>```bash
>autoload -Uz compinit
>zstyle ':completion:*' menu select
>fpath+=~/.zfunc
>compinit
>```
> This helps configure and enable a modified tab-completion for Garden.
> Now, autocompletion can work such as: ```garden-ai garden publish -g [TAB] [TAB]```

Now you're ready to start Gardening!
