import inspect
import json
import os
import pathlib
import tarfile
from tempfile import TemporaryDirectory
from typing import Optional

import docker  # type: ignore
import nbconvert  # type: ignore

JUPYTER_TOKEN = "791fb91ea2175a1bbf15e1c9606930ebdf6c5fe6a0c3d5bd"  # arbitrary valid token, safe b/c port is only exposed to localhost


def start_container_with_notebook(
    client: docker.DockerClient,
    path: pathlib.Path,
    base_image: str,
    platform: str = "linux/x86_64",
    mount: bool = True,
    pull: bool = True,
) -> docker.models.containers.Container:
    """
    Start a Docker container with a Jupyter Notebook server.

    The specified notebook file is mounted into the container at
    `/garden/notebook-name.ipynb`.

    Parameters:
    - client (docker.DockerClient): The Docker client object to interact
      with the Docker daemon.
    - path (pathlib.Path): The local path to the notebook.
    - base_image (str): The Docker image to be used as the base. It should
      have Jupyter pre-installed.
    - platform (str): Passed directly to docker sdk. Defaults to "linux/x86_64".
    - mount (bool): Whether the notebook should be mounted (True) or just
      copied in (False).
    - pull (bool): Whether the base_image exists locally or needs to be
      pulled down.

    Returns:
    - docker.models.containers.Container: The started container object.

    Note:
    - The Jupyter Notebook server inside the container runs on port 8888
      and is exposed to the host on the same port.
    - The token for accessing the notebook is still the JUPYTER_TOKEN variable.
    """
    if pull:
        client.images.pull(base_image, platform=platform)

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
            f"--NotebookApp.token={JUPYTER_TOKEN}",
            "--ip",
            "0.0.0.0",
            "--no-browser",
            "--allow-root",
        ],
        stdin_open=True,
        tty=True,
        remove=True,
    )

    if not mount:
        with TemporaryDirectory() as temp_dir:
            temp_dir_path = pathlib.Path(temp_dir)
            temp_tar_path = f"{str(temp_dir_path / path.name)}.tar"

            # when the notebook is added to tar, do not include parent dirs
            # by changing the workdir to the parent of the notebook
            curr_dir = os.getcwd()
            os.chdir(path.parent)

            with tarfile.open(temp_tar_path, mode="w") as tar:
                tar.add(path.name)
            with open(temp_tar_path, "rb") as f:
                stream = f.read()

            # explicitly return to previous directory
            os.chdir(curr_dir)

        container.put_archive("/garden", stream)

    return container


def build_notebook_session_image(
    client: docker.DockerClient,
    notebook_path: pathlib.Path,
    base_image: str,
    platform: str = "linux/x86_64",
    print_logs: bool = True,
) -> Optional[docker.models.images.Image]:
    """
    Build the docker image to register with Globus Compute locally.

    From the specified base image, this:
    - generates a plain python script from the user's notebook,
    - copies both to the container
    - RUNs the script for side effects, including writing the session.pkl and metadata.json

    This does not tag or push the image.

    Args:
        client: A Docker client instance to manage Docker resources.
        notebook_path: A Path object to the Jupyter notebook file.
        base_image: The name of the base Docker image to use.
        platform: The target platform for the Docker build (default is "linux/x86_64").
        print_logs: Flag to enable streaming build logs to the console (default is True).

    Returns:
        The docker `Image` object if the build succeeds, otherwise None.

    Raises:
        docker.errors.BuildError: If the Docker image build fails.
    """
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = pathlib.Path(temp_dir)

        # copy the notebook to the temp dir
        temp_notebook_path = temp_dir_path / notebook_path.name
        with temp_notebook_path.open("w+") as f:
            f.write(notebook_path.read_text())

        # convert notebook to sister script in temp dir
        exporter = nbconvert.PythonExporter()
        script_contents, _notebook_meta = exporter.from_filename(str(notebook_path))

        # append code to save the interpreter session and metadata
        # to the script contents
        import garden_ai.scripts.save_session_and_metadata  # type: ignore

        script_extra = inspect.getsource(garden_ai.scripts.save_session_and_metadata)
        script_contents += f"\n{script_extra}"

        script_path = temp_notebook_path.with_suffix(".py")
        script_path.write_text(script_contents)

        client.images.pull(base_image, platform=platform)

        # easier to grok than pure docker sdk equivalent (if one exists)
        dockerfile_content = f"""
        FROM {base_image}
        COPY {notebook_path.name} /garden/{notebook_path.name}
        COPY {script_path.name} /garden/{script_path.name}
        WORKDIR /garden
        RUN ipython {script_path.name}
        """
        dockerfile_path = temp_dir_path / "Dockerfile"
        with dockerfile_path.open("w") as f:
            f.write(dockerfile_content)

        # build the image and propagate logs
        print("Building image ...")
        stream = client.api.build(
            path=str(temp_dir_path), platform=platform, decode=True
        )
        image = None
        for chunk in stream:
            if "stream" in chunk:
                if print_logs:
                    print(chunk["stream"].strip())
            if "aux" in chunk and "ID" in chunk["aux"]:
                image_id = chunk["aux"]["ID"]
                image = client.images.get(image_id)
            if "error" in chunk or "errorDetail" in chunk:
                error_message = chunk.get(
                    "error", chunk.get("errorDetail", {}).get("message", "")
                )
                print("Build failed:", error_message)
                raise docker.errors.BuildError(reason=error_message, build_log=stream)

        return image


def extract_metadata_from_image(
    client: docker.DockerClient, image: docker.models.images.Image
) -> dict:
    """Load dict of metadata stored as `metadata.json` in the image.

    keys are the original function names, values are respective metadata dicts.

    see also: `garden_ai.scripts.save_session_and_metadata`
    """
    container_stdout = client.containers.run(
        image=image.id,
        entrypoint="/bin/sh",
        command=["-c", "cat /garden/metadata.json"],
        remove=True,
        detach=False,
    )
    raw_metadata = container_stdout.decode("utf-8")
    return json.loads(raw_metadata)


def push_image_to_public_repo(
    client: docker.DockerClient,
    image: docker.models.images.Image,
    repo_name: str,
    tag: str,
    print_logs: bool = True,
) -> str:
    """
    Tags and pushes a Docker image to a new public repository.

    Args:
        client: The Docker client instance.
        image: The Docker image to be pushed.
        repo_name: The name of the public repository to push the image (e.g., "username/myrepo").
        tag: The tag for the image.
        print_logs: Whether to stream logs from docker push to stdout (default: True)

    Returns:
        The full Docker Hub location of the pushed image (e.g. "docker.io/username/myrepo:tag" )
    """
    # Tag the image with the new repository name
    image.tag(repo_name, tag=tag)

    # push the image to the new repository and stream logs
    push_logs = client.images.push(repo_name, tag=tag, stream=True, decode=True)
    for log in push_logs:
        if "error" in log:
            # docker sdk doesn't throw its own exception if the repo doesn't
            # exist or the push fails, just records the error in logs
            error_message = log["error"]
            raise docker.errors.InvalidRepository(
                f"Error pushing image to {repo_name}:{tag} - {error_message}"
            )

        if print_logs:
            if "status" in log:
                if "progress" in log:
                    print(f"{log['status']} - {log['progress']}")
                else:
                    print(log["status"])

    return f"docker.io/{repo_name}:{tag}"
