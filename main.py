#!/usr/bin/env python3

from pathlib import Path
from typing import Any, Dict, List

import click
import dash
import dash_bootstrap_components as dbc

from layout import create_layout

def setup() -> dash.Dash:
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
    app.title = "Gradient Descent Analysis"
    app.layout = create_layout()

    return app


@click.command(
    help="""The last instance before failure. Everything you want to understand for gradient descent analysis.
    """
)
@click.option(
    "--port",
    help="Port of published web server",
    type=int,
    default=8050,
)
@click.option(
    "-d",
    "--debug",
    is_flag=True,
    help="Running dash board in debug mode. Beware data will be loaded twice.",
)
def main(
    port: int,
    debug: bool,
) -> None:
    """
    Display the damn epic Gradient Descent Analysis
    """

    # run dash app
    setup().run(port=port, debug=debug)


if __name__ == "__main__":
    main()
