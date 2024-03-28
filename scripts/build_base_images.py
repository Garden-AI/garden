#!/usr/bin/env python3
import docker  # type: ignore
from docker import DockerClient
from tempfile import TemporaryDirectory
from garden_ai.containers import process_docker_build_stream, DockerBuildFailure
import os


def build_garden_base_image(client: DockerClient, python_version) -> str:
    """Build and tag an image into the gardenai/base repository.

    Image is built from the official image for the given python version. Does
    not install any extras beside pandas, numpy and (jupyter) notebook.
    """
    dockerfile_content = f"""
    FROM --platform=linux/amd64 python:{python_version}
    WORKDIR /garden
    RUN pip install --no-cache-dir pandas numpy notebook
    """

    with TemporaryDirectory() as tmpdir:
        dockerfile_path = os.path.join(tmpdir, "Dockerfile")
        with open(dockerfile_path, "w") as dockerfile:
            dockerfile.write(dockerfile_content)
        image_tag = f"gardenai/base:python-{python_version}-base"
        stream = client.api.build(
            path=str(tmpdir),
            decode=True,
            tag=image_tag,
            rm=True,
            forcerm=True,
        )
        image = process_docker_build_stream(stream, client, DockerBuildFailure, True)
        return image.tags[0]


def build_flavor_image(client, python_version, flavor, extras):
    """Build a flavor image from the gardenai/base image with the same python version."""
    dockerfile_content = f"""
    FROM --platform=linux/amd64 gardenai/base:python-{python_version}-base
    """
    for extra in extras:
        dockerfile_content += f"\nRUN pip install --no-cache-dir {extra}"

    with TemporaryDirectory() as tmpdir:
        dockerfile_path = os.path.join(tmpdir, "Dockerfile")
        with open(dockerfile_path, "w") as dockerfile:
            dockerfile.write(dockerfile_content)
        image_tag = f"gardenai/base:python-{python_version}-{flavor}"
        stream = client.api.build(
            path=str(tmpdir),
            decode=True,
            tag=image_tag,
            rm=True,
            forcerm=True,
        )
        image = process_docker_build_stream(stream, client, DockerBuildFailure, True)
        return image.tags[0]


def push_to_dockerhub(client: DockerClient, image_tag: str):
    push_logs = client.images.push(repository=image_tag, stream=True, decode=True)
    for log in push_logs:
        if "error" in log:
            error_message = log["error"]
            raise docker.errors.InvalidRepository(
                f"Error pushing image to {image_tag} - {error_message}"
            )
        if "status" in log:
            if "progress" in log:
                print(f"{log['status']} - {log['progress']}")
            else:
                print(log["status"])

    return image_tag


def remove_local_images(client: docker.DockerClient):
    for image in client.images.list(name="gardenai/base"):
        try:
            print(f"Removing image: {list(image.tags)}")
            client.images.remove(image.id)
        except docker.errors.APIError as e:
            print(f"Failed to remove image {image.id}: {e}")
    return


FLAVOR_EXTRAS = {
    "tensorflow": ["tensorflow"],
    "sklearn": [
        "joblib",
        "scipy",
        "scikit-learn",
        "scikit-learn-extra",
        "scikit-optimize",
    ],
    "torch": [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
    ],
}
FLAVOR_EXTRAS["all"] = [
    pkg for (flavor, extras) in FLAVOR_EXTRAS.items() for pkg in extras
]
FLAVOR_EXTRAS["wisconsin"] = ["pymatgen", "mastml", "madml"]

if __name__ == "__main__":
    client = docker.from_env()

    for version in ["3.8", "3.9", "3.10", "3.11"]:
        # build base variant first
        no_extras = build_garden_base_image(client, version)
        push_to_dockerhub(client, no_extras)

        for flavor, extras in FLAVOR_EXTRAS.items():
            # build flavor images from base
            print(f"building gardenai/base:python-{version}-{flavor}")
            image_tag = build_flavor_image(client, version, flavor, extras)
            push_to_dockerhub(client, image_tag)

        # not needed if you have enough disk space to build all the images
        remove_local_images(client)
