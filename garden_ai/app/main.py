import logging
import typer
import garden_ai.app.create as create

logger = logging.getLogger()

app = typer.Typer()

# nb: subcommands are mini typer apps in their own right
# app.add_typer(create.app, name="create")
#
app.command(name="create", no_args_is_help=True)(create.create_garden)


@app.callback()
def help_info():
    """
    🌱 Hello, Garden 🌱

    I'm some help text!
    """
    # TODO this is where --version, --verbose etc logic should go
    pass
