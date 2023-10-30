import datetime
import inspect
import json
import pathlib
import textwrap
from tempfile import TemporaryDirectory
from typing import Optional

import docker  # type: ignore
import nbconvert  # type: ignore

JUPYTER_TOKEN = "791fb91ea2175a1bbf15e1c9606930ebdf6c5fe6a0c3d5bd"  # arbitrary valid token, safe b/c port is only exposed to localhost


def start_container_with_notebook(
    client: docker.DockerClient,
    path: pathlib.Path,
    base_image: str,
    platform="linux/x86_64",
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
      have Jupyter Notebook pre-installed.
    - platform (str): Passed directly to docker sdk. Defaults to "linux/x86_64".

    Returns:
    - docker.models.containers.Container: The started container object.

    Note:
    - The Jupyter Notebook server inside the container runs on port 8888
      and is exposed to the host on the same port.
    - The token for accessing the notebook is still the JUPYTER_TOKEN variable.
    """
    client.images.pull(base_image, platform=platform)
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
    client: docker.DockerClient,
    notebook_path: pathlib.Path,
    base_image: str,
    platform="linux/x86_64",
    print_logs=True,
) -> Optional[docker.models.images.Image]:
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = pathlib.Path(temp_dir)

        # copy the notebook to the temp dir
        temp_notebook_path = temp_dir_path / notebook_path.name
        with temp_notebook_path.open("w+") as f:
            f.write(notebook_path.read_text())

        # convert notebook to sister script in temp dir
        exporter = nbconvert.PythonExporter()
        script_contents, _notebook_meta = exporter.from_filename(str(notebook_path))
        script_contents += "\nimport dill\ndill.dump_session('session.pkl')\n"

        script_path = temp_notebook_path.with_suffix(".py")
        script_path.write_text(script_contents)

        client.images.pull(base_image, platform=platform)

        # easier to grok than pure docker sdk equivalent
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

        # notebook name + timestamp seemed like a good balance of
        # human-readability and uniqueness for a tag
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        image_name = f"{notebook_path.stem}-{timestamp}"
        # build the image and propagate logs
        try:
            print("Building image ...")
            image, logs = client.images.build(
                path=str(temp_dir_path),
                tag=image_name,
                platform=platform,
            )
        except docker.errors.BuildError as err:
            logs = err.build_log
            raise
        finally:
            if print_logs:
                for log in logs:
                    print(log.get("stream", "").strip())
        return image


def extract_metadata_from_image(
    client: docker.DockerClient, image: docker.models.images.Image
):
    def _print_metadata_from_session() -> None:
        import json

        import dill  # type: ignore

        dill.load_session("session.pkl")

        decorated_fns = []
        global_vars = list(globals().values())

        for obj in global_vars:
            if hasattr(obj, "_pipeline_meta") and hasattr(obj, "_model_connectors"):
                decorated_fns.append(obj)

        if len(decorated_fns) == 0:
            raise ValueError("No functions marked with garden decorator.")

        total_meta = {}

        for marked in decorated_fns:
            key_name = marked._pipeline_meta["short_name"]
            connector_key = f"{key_name}.connectors"

            total_meta[key_name] = marked._pipeline_meta
            total_meta[connector_key] = [
                connector.metadata for connector in marked._model_connectors
            ]

        print(json.dumps(total_meta))  # stdout is captured
        return

    # build a python command to define and immediately call the function above
    function_source = textwrap.dedent(inspect.getsource(_print_metadata_from_session))
    command = f'python -c "{function_source}\n_print_metadata_from_session()"'

    # spin up the container
    container = client.containers.run(
        image=image.id, command="/bin/sh", detach=True, tty=True, remove=True
    )

    try:
        # execute the python command in the container
        exit_code, output = container.exec_run(command)

        if exit_code != 0:
            raise RuntimeError(
                f"Command failed with code {exit_code}: {output.decode()}"
            )

        # output is the JSON string printed from `_print_metadata_from_session`.
        return json.loads(output.decode())

    finally:
        container.remove(force=True)
