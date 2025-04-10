from typer import Typer

from uta_ahp.model.base import Model

app = Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
)

Model.register_commands(app)
