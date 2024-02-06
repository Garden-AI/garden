import typer

from garden_ai.containers import (
    get_docker_client,
    DockerStartFailure,
    cleanup_dangling_images,
)
from garden_ai.app.console import print_err
from garden_ai.constants import GardenConstants

docker_app = typer.Typer(name="docker", no_args_is_help=True)

ASCII_FLOWER = """
     /\^/`\\
    | \/   |
    | |    |
    \ \    /
     '\\\\//'
       ||
       ||
       ||
       ||  ,
   |\  ||  |\\
   | | ||  | |
   | | || / /
    \ \||/ /
jgs  `\\//`
    ^^^^^^^^"""


@docker_app.command()
def check(
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Check if Garden can access Docker on your computer.

    If Garden can't access Docker and it's not clear what the problem is,
    using --verbose will print out a full stack trace.
    """
    try:
        client = get_docker_client()
        client.ping()
        typer.echo("Docker is running and accessible. Happy Gardening!")
        typer.secho(ASCII_FLOWER, fg=typer.colors.GREEN)
    except DockerStartFailure as e:
        print_err("Garden can't access Docker on your computer.")
        if e.helpful_explanation:
            print_err(e.helpful_explanation)
        else:
            print_err(
                "This doesn't look like one of the typical error cases. Printing error from Docker:"
            )
            typer.echo(e.original_exception)
        if verbose:
            raise e


@docker_app.command()
def prune(
    keep_base: bool = typer.Option(
        False,
        "--keep-base",
        help="If enabled, keep official gardenai/base images and only remove custom user images (e.g. those created by publishing)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="If enabled, just print the tags of images that would be pruned.",
    ),
    remove_dangling: bool = typer.Option(
        False,
        "--remove-dangling",
        help="Remove any dangling images. This includes dangling images that may not be Garden-related.",
    ),
):
    """Remove Garden-related Docker images, freeing up disk space."""
    client = get_docker_client()
    if remove_dangling:
        if dry_run:
            for image in client.images.list(filters={"dangling": True}):
                print(f"Would remove dangling image {image.short_id}")
        else:
            cleanup_dangling_images(client)

    prefixes = ["gardenai/custom", GardenConstants.GARDEN_ECR_REPO]
    if not keep_base:
        prefixes += ["gardenai/base"]

    for image in client.images.list():
        for tag in image.tags:
            if tag.startswith(tuple(prefixes)):
                if dry_run:
                    print(f"Would remove {image.short_id} ({tag})")
                else:
                    print(f"Removing {image.short_id} ({tag})")
                    client.images.remove(tag)
