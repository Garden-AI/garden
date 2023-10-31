import pathlib
from garden_ai.app.console import console

import docker  # type: ignore

JUPYTER_TOKEN = "791fb91ea2175a1bbf15e1c9606930ebdf6c5fe6a0c3d5bd"  # arbitrary valid token, safe b/c port is only exposed to localhost


def start_container_with_notebook(
    path: pathlib.Path, client: docker.DockerClient, base_image: str
) -> docker.models.containers.Container:
    """
    Start a Docker container with a Jupyter Notebook server.

    The specified notebook file is mounted into the container at
    `/garden/notebook-name.ipynb`.

    Parameters:
    - path (pathlib.Path): The local path to the notebook.
    - base_image (str): The Docker image to be used as the base. It should
      have Jupyter Notebook pre-installed.
    - client (docker.DockerClient): The Docker client object to interact
      with the Docker daemon.

    Returns:
    - docker.models.containers.Container: The started container object.

    Note:
    - The Jupyter Notebook server inside the container runs on port 8888
      and is exposed to the host on the same port.
    - The token for accessing the notebook is still the JUPYTER_TOKEN variable.
    """

    with console.status(f"[bold green] Pulling image: {base_image}"):
        client.images.pull(base_image, platform="linux/x86_64")
    container = client.containers.run(
        base_image,
        platform="linux/x86_64",
        detach=True,
        ports={"8888/tcp": 8888},
        volumes={str(path): {"bind": f"/garden/{path.name}", "mode": "rw"}},
        entrypoint=[
            "jupyter",
            "notebook",
            "--notebook-dir=/garden",
            f"--ServerApp.token={JUPYTER_TOKEN}",
            "--ip",
            "0.0.0.0",
            "--no-browser",
            "--allow-root",
        ],
        stdin_open=True,
        tty=True,
        remove=True,
    )
    return container
