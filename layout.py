from collections import OrderedDict
from typing import Any, Dict, List

import dash_bootstrap_components as dbc
from dash import dcc, html

from factory import FunctionFactory

function_ids: List[str] = FunctionFactory.available_functions()


def create_graph_card(
    header_id: str, header_val: str, graph_id: str, graph_height: int, body_height: int
) -> dbc.Card:
    card_content = [
        dbc.CardHeader(header_val, id=header_id),
        dbc.CardBody(
            [dcc.Graph(id=graph_id, className="h-100")],
            style={"height": graph_height},
        ),
    ]
    return dbc.Card(card_content, body=True, style={"height": body_height})


def create_title() -> html.H1:
    return html.H1("Dashboard", className="text-center")


def create_interactive_board() -> List[dbc.Col]:
    board_elements: List[dbc.Col] = [
        dbc.Col(
            dcc.Dropdown(
                id="function-dropdown",
                options=function_ids,
                value=function_ids[0],
            ),
            width=3,
        ),
        dbc.Col(
            dcc.Dropdown(
                id="epic-all-or-single-object-view",
                options=[
                    {"label": "Show all", "value": "all"},
                    {"label": "Show only selected", "value": "selected"},
                ],
                value="all",
            ),
            width=2,
        ),
        dbc.Col(
            dcc.Input(
                id="num-contours-input",
                type="number",
                min=1,
                max=50,
                value=10,
                step=1,
                placeholder="Number of contours",
            ),
            width=2,
        ),
    ]

    return board_elements


def create_first_row_graphs() -> List[dbc.Col]:
    first_row_graphs: List[dbc.Col] = [
        dbc.Col(
            create_graph_card("view-3d-header", "3D View", "view-3d-graph", 700, 900),
            width=6,
        ),
        dbc.Col(
            create_graph_card("top-view-header", "Top View", "top-view-graph", 700, 900),
            width=6,
        ),
    ]
    return first_row_graphs


def create_second_row_graphs() -> List[dbc.Col]:
    second_row_graphs: List[dbc.Col] = [
        dbc.Col(
            create_graph_card("view-2d-header", "Side View", "view-2d-graph", 700, 900),
            width=6,
        ),
        dbc.Col(
            create_graph_card("result-view-header", "Result View", "result-view-graph", 700, 900),
            width=6,
        ),
    ]
    return second_row_graphs


def create_layout() -> dbc.Container:
    print("Layout created")
    layout = dbc.Container(
        [
            dbc.Row(
                create_title(),
            ),
            dbc.Row(
                create_interactive_board(),
            ),
            dbc.Row(
                create_first_row_graphs(),
            ),
            dbc.Row(
                create_second_row_graphs(),
            ),
        ],
        # style={"height": "100vh"},
        fluid=True,
    )
    return layout
