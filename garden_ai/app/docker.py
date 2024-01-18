import typer

from garden_ai.containers import get_docker_client, DockerStartFailure
from garden_ai.app.console import print_err

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
