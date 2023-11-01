import pathlib
import subprocess
from garden_ai.app.console import console

import docker  # type: ignore

JUPYTER_TOKEN = "791fb91ea2175a1bbf15e1c9606930ebdf6c5fe6a0c3d5bd"  # arbitrary valid token, safe b/c port is only exposed to localhost


def start_container_with_notebook(
    path: pathlib.Path, client: docker.DockerClient, base_image: str, mount: bool = True
) -> docker.models.containers.Container:
    """
    Start a Docker container with a Jupyter Notebook server.

    The specified notebook file is mounted into the container at
    `/garden/notebook-name.ipynb`.

    Parameters:
    - path (pathlib.Path): The local path to the notebook.
    - client (docker.DockerClient): The Docker client object to interact
      with the Docker daemon.
    - base_image (str): The Docker image to be used as the base. It should
      have Jupyter Notebook pre-installed.
    - mount (bool): Whether the notebook should be mounted (True) or just
      copied in (False)

    Returns:
    - docker.models.containers.Container: The started container object.

    Note:
    - The Jupyter Notebook server inside the container runs on port 8888
      and is exposed to the host on the same port.
    - The token for accessing the notebook is still the JUPYTER_TOKEN variable.
    """
    # first check if the image is present, otherwise attempt to pull
    try:
        client.images.get(base_image)
    except docker.errors.ImageNotFound:
        with console.status(f"[bold green] Pulling image: {base_image}"):
            client.images.pull(base_image, platform="linux/x86_64")

    if mount:
        volumes = {str(path): {"bind": f"/garden/{path.name}", "mode": "rw"}}
    else:
        volumes = {}

    container = client.containers.run(
        base_image,
        platform="linux/x86_64",
        detach=True,
        ports={"8888/tcp": 8888},
        volumes=volumes,
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

    # the equivalent docker-py function only allows transfer of tar archives
    # apparently, this is how the API works under-the-hood
    # but I think the following is far clearer than messing with byte streams
    if not mount:
        subprocess.run(["docker", "cp", str(path), f"{container.name}:/garden"])

    return container
