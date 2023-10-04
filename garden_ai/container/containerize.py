import pathlib
import subprocess
import time
import webbrowser

from tempfile import TemporaryDirectory
from threading import Thread
from typing import List, Optional

IMAGE_NAME = "gardenai/base"
CONTAINER_NAME = "garden_ai"
JUPYTER_TOKEN = "791fb91ea2175a1bbf15e1c9606930ebdf6c5fe6a0c3d5bd"  # arbitrary valid token, safe b/c port is only exposed to localhost


def build_container(image_name: str = IMAGE_NAME, jupyter: bool = False) -> None:
    with TemporaryDirectory() as tmpdir:
        tmpdir = tmpdir.replace("\\", "/")  # Windows compatibility

        with open(f"{tmpdir}/Dockerfile", "w") as dockerfile:
            dockerfile.writelines(
                [
                    "FROM python:3.10-slim\n",  # solution to hard-coding global Python version might be prudent (miniconda install link must match)
                    "WORKDIR /garden\n",
                    "RUN apt-get update && ",
                    "apt-get install -y git && ",
                    "apt-get install -y wget && ",
                    "apt-get install -y dos2unix && ",
                    "apt-get clean && ",
                    "rm -rf /var/lib/apt/lists/*\n",
                    "RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -O ~/miniconda.sh && ",
                    "/bin/bash ~/miniconda.sh -b -p /opt/conda\n",
                    "ENV PATH=/opt/conda/bin:$PATH\n",
                    "RUN conda create -y --name env\n",
                    'RUN pip install "dill==0.3.5.1"\n',
                ]
            )
            if jupyter:
                image_name = (
                    f"{image_name}-jupyter" if image_name == IMAGE_NAME else image_name
                )
                dockerfile.writelines(
                    [
                        "RUN pip install jupyter notebook\n",
                        f'ENTRYPOINT ["jupyter", "notebook", "--notebook-dir=/garden", "--ServerApp.token={JUPYTER_TOKEN}", '
                        '"--ip", "0.0.0.0", "--no-browser", "--allow-root"]',
                    ]
                )
            else:
                dockerfile.write('ENTRYPOINT ["/bin/bash"]')

        subprocess.run(
            [
                "docker",
                "buildx",
                "build",
                "--platform=linux/x86_64",
                "-t",
                image_name,
                tmpdir,
            ]
        )


def start_container(
    image_name: str = IMAGE_NAME,
    container_name: str = CONTAINER_NAME,
    jupyter: bool = False,
    entrypoint: Optional[str] = None,
    cleanup: bool = False,
    args: Optional[List[str]] = None,
) -> None:
    if jupyter and entrypoint:
        raise NotImplementedError(
            "This combination of arguments results in undefined behavior, and is unsupported."
        )

    if args is None:
        args = []

    image_name = (
        f"{image_name}-jupyter" if jupyter and image_name == IMAGE_NAME else image_name
    )

    run_command = [
        "docker",
        "run",
        "-it",
        "--rm",
        "--platform",
        "linux/x86_64",
        "--name",
        container_name,
        image_name,
    ]
    run_command += args

    if jupyter:
        idx = run_command.index(image_name)
        # yes, this is the correct order with which to insert
        run_command.insert(idx, "127.0.0.1:8888:8888/tcp")
        run_command.insert(idx, "-p")
        run_command.insert(idx, f"{pathlib.Path().resolve()}:/garden")
        run_command.insert(idx, "-v")

        t = Thread(
            target=lambda x: not time.sleep(2) and webbrowser.open_new_tab(x),  # type: ignore
            args=[f"http://127.0.0.1:8888/tree?token={JUPYTER_TOKEN}"],
        )
        t.start()

    if entrypoint is not None:
        idx = run_command.index(image_name)
        run_command.insert(idx, entrypoint)
        run_command.insert(idx, "--entrypoint")

    subprocess.run(run_command)
    if cleanup:
        subprocess.run(["docker", "rmi", image_name])


def garden_pipeline(func):
    def garden_target(*args, **kwargs):
        return func(*args, **kwargs)

    garden_target._check = True
    return garden_target  # returns func back, but with `__name__ == garden_target` and a _check attr
