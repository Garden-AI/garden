import logging
import typer
from garden_ai.app.garden import garden_app
from garden_ai.app.pipeline import pipeline_app
from garden_ai.app.model import model_app

logger = logging.getLogger()

app = typer.Typer(no_args_is_help=True)


# nb: subcommands are mini typer apps in their own right
app.add_typer(garden_app)
app.add_typer(pipeline_app)
app.add_typer(model_app)


@app.callback()
def help_info():
    """
    🌱 Hello, Garden 🌱

    I'm some help text!
    """
    pass
