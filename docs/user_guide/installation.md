# Prerequisites

Garden makes your AI/ML work reproducible by using [Docker](https://www.docker.com/) containers to fully encapsulate your code and its dependencies. This helps ensure consistency across different machines, making it easier for anyone to reproduce your results. Before you get started with Garden, there are two key Docker-specific prerequisites:

### Docker Desktop

1. **Install Docker Desktop:** To use Garden, you need to have Docker Desktop installed on your system. You can download it from the [official Docker website](https://www.docker.com/products/docker-desktop). Follow the installation instructions specific to your operating system (Windows, macOS, or Linux).

2. **Verify Docker Installation:** Once installed, you can verify that Docker Desktop is running correctly by opening a terminal or command prompt and typing:
```bash
docker --version
```
This command should return the version of Docker installed on your system.

### Docker Hub Account

1. **Create a Docker Hub Account:** Garden uses Docker Hub, a cloud-based repository for Docker images, to store and manage containers. You need to create a free account on [Docker Hub](https://hub.docker.com/signup) if you don't already have one.

2. **Create a Public Repository:** After setting up your Docker Hub account, create a public repository where you can push your Docker images. This repository will be used to store and share the Docker images of your projects.

3. **Configure Docker Credentials:** Make sure to configure your Docker credentials on your system. This allows `garden-ai` to push and pull images to and from Docker Hub on your behalf. You can do this by running:
```bash
docker login
```
and entering your Docker Hub username and password when prompted.
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
