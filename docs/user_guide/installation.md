# Installation

Installing the `garden-ai` python package contains both our SDK and our CLI. We _highly_ recommend installing the CLI with a virtual environment, using venv or conda.

Below are instructions for installing with venv.

```bash
mkdir my_garden_project
cd my_garden_project
python3 -m venv garden-env
source myenv/bin/activate
pip install garden-ai
```

To verify your installation, you can check the version of Garden:

```bash
garden-ai --version
```

You should see the version of your installed Garden package printed to the console.

Garden also supports tab completion in the CLI to show options of local models, pipelines, and gardens.
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

> [!NOTE] Note
> Especially when installing with additional flavors, the CLI might be very slow to respond the first time it is invoked (or the first time the SDK is imported), but it should be much snappier every time after the first.
