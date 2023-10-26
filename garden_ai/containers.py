import datetime
import pathlib
from tempfile import TemporaryDirectory

import docker  # type: ignore
import nbconvert  # type: ignore

JUPYTER_TOKEN = "791fb91ea2175a1bbf15e1c9606930ebdf6c5fe6a0c3d5bd"  # arbitrary valid token, safe b/c port is only exposed to localhost


def start_container_with_notebook(
    client: docker.DockerClient, path: pathlib.Path, base_image: str
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


def build_notebook_session_image(
    client: docker.DockerClient, notebook_path: pathlib.Path, base_image: str
) -> str:
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = pathlib.Path(temp_dir)

        # copy the notebook to the temp dir
        temp_notebook_path = temp_dir_path / notebook_path.name
        with temp_notebook_path.open("w+") as f:
            f.write(notebook_path.read_text())

        # convert notebook to sister script in temp dir
        exporter = nbconvert.PythonExporter()
        script_contents, _notebook_meta = exporter.from_filename(notebook_path)
        script_contents += "\nimport dill\ndill.dump_session('session.pkl')\n"

        script_path = temp_notebook_path.with_suffix(".py")
        script_path.write_text(script_contents)

        client.images.pull(base_image, platform="linux/x86_64")

        # easier to grok than pure docker sdk equivalent
        dockerfile_content = f"""
        FROM {base_image}
        COPY {notebook_path.name} /garden/{notebook_path.name}
        COPY {script_path.name} /garden/{script_path.name}
        WORKDIR /garden
        RUN python -m {script_path.stem}
        """
        dockerfile_path = temp_dir_path / "Dockerfile"
        with dockerfile_path.open("w") as f:
            f.write(dockerfile_content)

        # build the new image
        # notebook name + timestamp seemed like a good balance of
        # human-readability and uniqueness for a tag
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        image_name = f"{notebook_path.stem}-{timestamp}"
        # TODO propagate errors from RUN
        image, _ = client.images.build(
            path=str(temp_dir_path), tag=image_name, platform="linux/x86_64"
        )

        return image_name
