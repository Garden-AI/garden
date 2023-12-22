# Prerequisites

Garden makes your AI/ML work reproducible by using [Docker](https://www.docker.com/) containers to fully encapsulate your code and its dependencies. This helps ensure consistency across different machines, making it easier for anyone to reproduce your results.

### Docker

1. **Install Docker:** To publish with Garden, you need to have Docker installed and running on your system. An easy way to install and manage Docker is through [Docker desktop](https://www.docker.com/products/docker-desktop). Follow the installation instructions specific to your operating system (Windows, macOS, or Linux).

2. **Verify Docker Installation:** Once installed, you can verify that Docker Desktop is running correctly by opening a terminal or command prompt and typing:
```bash
docker --version
```
This command should return the version of Docker installed on your system.

# Installation

Installing the `garden-ai` python package contains both our SDK and our CLI. We _highly_ recommend installing the CLI with a virtual environment, using venv or conda. [pipx](https://pipx.pypa.io/stable/) works well too.

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
