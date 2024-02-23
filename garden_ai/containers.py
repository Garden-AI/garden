import inspect
import json
import os
import io
import pathlib
import tarfile
import functools
import datetime
from tempfile import TemporaryDirectory
from typing import Optional, Iterator, Union, Type

import docker  # type: ignore

from garden_ai.constants import GardenConstants


class DockerStartFailure(Exception):
    """
    Raised when Garden can't access Docker.
    """

    def __init__(self, original_exception, explanation=None):
        self.original_exception = original_exception
        self.helpful_explanation = explanation
        super().__init__(f"Docker failed to start: {original_exception}")


class DockerPreBuildFailure(docker.errors.BuildError):
    """
    Raised when Docker fails at the pre-build stage.
    That's when we are building the base container with the user's custom requirements.
    """


class DockerBuildFailure(docker.errors.BuildError):
    """
    Raised when Docker fails during a container build.
    That's when we are building the publishable container with the user's custom code.
    """


def handle_docker_errors(func):
    """
    This decorator catches common classes of Docker errors and sends recommendations for how to fix them up the callstack.
    If the user can't run Docker at all, it raises a DockerStartFailure.
    If the user runs into a BuildError, it tries to remove any dangling intermediate images before re-raising the exception.
    Otherwise the normal Docker exception is still raised.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except docker.errors.DockerException as e:
            error_message = str(e)
            # We only handle cases that happen when we can't even connect to Docker
            if "Error while fetching server API version" not in error_message:
                # So if it's not that kind, just raise the original exception
                raise e

            explanation = None
            # If the daemon isn't running, Linux says "Connection refused". MacOS says "No such file or directory"
            if (
                "ConnectionRefusedError" in error_message
                or "FileNotFoundError" in error_message
            ):
                explanation = "Could not connect to your local Docker daemon. Double check that Docker is running."

            # If the daemon is running but you don't have permissions, Linux says "Permission denied".
            # This is less common on MacOS.
            elif "PermissionError" in error_message:
                explanation = (
                    "It looks like your current user does not have permissions to use Docker. "
                    "Try adding your user to your OS's Docker group with the following command: "
                    "sudo usermod -aG docker ${whoami}"
                )

            raise DockerStartFailure(e, explanation)

    return wrapper


def cleanup_dangling_images(client: docker.DockerClient):
    """Naively remove any local images without a name or tag.

    This can fail if e.g. the dangling image is still referenced by a container.
    """
    for image in client.images.list(filters={"dangling": True}):
        try:
            client.images.remove(image.id)
        except docker.errors.APIError as e:
            print(f"Failed to remove dangling image {image.id}: {e}")
    return


@handle_docker_errors
def get_docker_client() -> docker.DockerClient:
    return docker.from_env()


@handle_docker_errors
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
            "--NotebookApp.token=''",
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


@handle_docker_errors
def build_notebook_session_image(
    client: docker.DockerClient,
    notebook_path: pathlib.Path,
    base_image: str,
    platform: str = "linux/x86_64",
    print_logs: bool = True,
    env_vars: dict = None,
) -> Optional[docker.models.images.Image]:
    """
    Build the docker image to register with Globus Compute locally.

    From the specified base image, this:
    - generates a plain python script from the user's notebook,
    - copies both to the container
    - RUNs the script for side effects, including writing the session.pkl and metadata.json

    This tags but does not push the image.

    Args:
        client: A Docker client instance to manage Docker resources.
        notebook_path: A Path object to the Jupyter notebook file.
        base_image: The name of the base Docker image to use.
        platform: The target platform for the Docker build (default is "linux/x86_64").
        print_logs: Flag to enable streaming build logs to the console (default is True).
        env_vars: dict of env variables to set in the built image (default {"GARDEN_SKIP_TESTS": True})

    Returns:
        The docker `Image` object if the build succeeds, otherwise None.

    Raises:
        DockerBuildFailure: If the Docker image build fails.
    """
    # build lines of 'ENV VAR=value\n' commands for dockerfile below
    env_vars = env_vars or {}
    if "GARDEN_SKIP_TESTS" not in env_vars:
        env_vars["GARDEN_SKIP_TESTS"] = True
    env_commands = "\n".join(f"ENV {k}={v}" for (k, v) in env_vars.items())

    with TemporaryDirectory() as temp_dir:
        temp_dir_path = pathlib.Path(temp_dir)

        # copy the notebook to the temp dir
        temp_notebook_path = temp_dir_path / notebook_path.name
        with temp_notebook_path.open("w+") as f:
            f.write(notebook_path.read_text())

        import nbconvert  # lazy import to speed up garden cold start

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

        # easier to grok than pure docker sdk equivalent (if one exists)
        dockerfile_content = f"""
        FROM {base_image}
        COPY {notebook_path.name} /garden/{notebook_path.name}
        COPY {script_path.name} /garden/{script_path.name}
        WORKDIR /garden
        {env_commands}
        RUN ipython {script_path.name}
        """
        dockerfile_path = temp_dir_path / "Dockerfile"
        with dockerfile_path.open("w") as f:
            f.write(dockerfile_content)

        # tag into local gardenai/custom repo for simpler cleanup later
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        full_tag = f"gardenai/custom:final-{timestamp}"
        # build/tag the image and propagate logs
        print("Building image ...")
        stream = client.api.build(
            path=str(temp_dir_path),
            platform=platform,
            decode=True,
            tag=full_tag,
            rm=True,
            forcerm=True,  # clean up unsuccessful builds too
        )
        image = process_docker_build_stream(
            stream, client, DockerBuildFailure, print_logs
        )
        return image


@handle_docker_errors
def extract_metadata_from_image(
    client: docker.DockerClient, image: docker.models.images.Image
) -> dict:
    """Load dict of metadata stored as `metadata.json` in the image.

    keys are the original function names, values are respective metadata dicts.

    see also: `garden_ai.scripts.save_session_and_metadata`
    """
    file_path = "/garden/metadata.json"
    container = client.containers.create(image.id, command="tail -f /dev/null")
    container.start()
    try:
        # get_archive returns a stream of the selected file as a tarball
        stream, _ = container.get_archive(file_path)
        file_obj = io.BytesIO(b"".join(stream))
        tar = tarfile.open(fileobj=file_obj)

        # There is only one file in the tarball
        member = tar.getmembers()[0]
        member_file_obj = tar.extractfile(member)
        if not member_file_obj:
            raise FileNotFoundError(f"Could not find {file_path} in image {image.id}")
        decoded_file_contents = member_file_obj.read().decode("utf-8")
    finally:
        container.stop()
        container.remove()

    return json.loads(decoded_file_contents)


@handle_docker_errors
def push_image_to_public_repo(
    client: docker.DockerClient,
    image: docker.models.images.Image,
    auth_config: dict,
    print_logs: bool = True,
) -> str:
    """
    Push and tag a Docker image to Garden ECR

    Args:
        client: The Docker client instance.
        image: The Docker image to be pushed.
        tag: The tag for the image (not including the repo).
        auth_config: auth dict for push to ECR
        print_logs: Whether to stream logs from docker push to stdout (default: True)

    Returns:
        The public location of the successfully pushed image (i.e. "{GARDEN_ECR_REPO}:{tag}")

    Raises:
        docker.errors.InvalidRepository: if the push fails
    """
    repo_name = GardenConstants.GARDEN_ECR_REPO
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")
    full_tag = f"{repo_name}:pushed-{timestamp}"
    image.tag(full_tag)

    # push the image to the new repository and stream logs
    push_logs = client.images.push(
        full_tag, auth_config=auth_config, stream=True, decode=True
    )
    for log in push_logs:
        if "error" in log:
            # docker sdk doesn't throw its own exception if the repo doesn't
            # exist or the push fails, just records the error in logs
            error_message = log["error"]
            raise docker.errors.InvalidRepository(
                f"Error pushing image to {full_tag} - {error_message}"
            )

        if print_logs:
            if "status" in log:
                if "progress" in log:
                    print(f"{log['status']} - {log['progress']}")
                else:
                    print(log["status"])

    return full_tag


@handle_docker_errors
def build_image_with_dependencies(
    client: docker.DockerClient,
    base_image: str,
    requirements_path: Optional[pathlib.Path] = None,
    platform: str = "linux/x86_64",
    print_logs: bool = True,
    pull: bool = True,
) -> str:
    """
    Build a "pre-baked" image from the base image with optional additional dependencies installed.

    This always installs at least the latest version of garden-ai.

    If included, the dependencies can be either a pip requirements.txt or a conda environment.yml file.

    Args:
        client: A Docker client instance to manage Docker resources.
        base_image: The name of the base Docker image to use.
        requirements_path: Optional; A Path object to the requirements.txt or environment.yml file.
        platform: The target platform for the Docker build (default is "linux/x86_64").
        print_logs: Enable streaming build logs to the console (default is True).
        pull: Whether to pull the base image before building on top of it (default is True).

    Returns:
        The ID of the freshly-built Docker image

    Raises:
        DockerPreBuildFailure: If the build fails.
    """

    # always install garden first
    dockerfile_content = f"""
    FROM {base_image}
    RUN pip install garden-ai
    """
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = pathlib.Path(temp_dir)

        # append to Dockerfile based on whether dependencies are provided
        if requirements_path is not None:
            temp_dependencies_path = temp_dir_path / requirements_path.name
            with temp_dependencies_path.open("w+") as f:
                f.write(requirements_path.read_text())

            if requirements_path.suffix == ".txt":  # pip
                dockerfile_content += f"""
                COPY {requirements_path.name} /garden/{requirements_path.name}
                RUN pip install -r /garden/{requirements_path.name}
                """
            else:  # conda
                # nb: installs directly into base environment instead of isolating from image's already-installed packages
                dockerfile_content += f"""
                COPY {requirements_path.name} /garden/{requirements_path.name}
                RUN conda env update --name base --file /garden/{requirements_path.name} && conda clean -afy
                """

        dockerfile_path = temp_dir_path / "Dockerfile"
        with dockerfile_path.open("w") as f:
            f.write(dockerfile_content)

        print("Preparing image ...")
        # tag into local-only repo
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        full_tag = f"gardenai/custom:base-{timestamp}"
        # Build and tag the image
        stream = client.api.build(
            path=str(temp_dir_path),
            platform=platform,
            decode=True,
            pull=pull,
            tag=full_tag,
            rm=True,
            forcerm=True,  # clean up unsuccessful builds too
        )
        image = process_docker_build_stream(
            stream, client, DockerPreBuildFailure, print_logs
        )
        return image.id


def process_docker_build_stream(
    stream: Iterator[dict],
    client: docker.DockerClient,
    exception_class: Union[Type[DockerBuildFailure], Type[DockerPreBuildFailure]],
    print_logs: bool,
) -> docker.models.images.Image:
    """
    Process the stream from a Docker build, handle errors, and return the image.

    Args:
        stream: The stream from the Docker API build call.
        client: The Docker client instance.
        exception_class: The exception class to raise if the build fails.
        print_logs: Whether to print the build logs to stdout.

    Returns:
        The image ID of the successfully built Docker image.

    Raises:
        exception of type `exception_class`: If the build fails.
    """
    image_id = None
    build_log = []
    for chunk in stream:
        if "stream" in chunk:
            stripped_line = chunk["stream"].strip()
            build_log.append(stripped_line)
            if print_logs:
                print(stripped_line)
        if "aux" in chunk and "ID" in chunk["aux"]:
            image_id = chunk["aux"]["ID"]
            image = client.images.get(image_id)
        if "error" in chunk or "errorDetail" in chunk:
            error_message = chunk.get(
                "error", chunk.get("errorDetail", {}).get("message", "")
            )
            raise exception_class(reason=error_message, build_log=build_log)

    if image is None:
        raise exception_class(
            reason="Could not find image ID in build logs", build_log=build_log
        )

    return image
