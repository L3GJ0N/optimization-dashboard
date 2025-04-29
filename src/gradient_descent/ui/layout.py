import dash_bootstrap_components as dbc
from dash import dcc, html

from gradient_descent.utils.factory import FunctionFactory

function_ids: list[str] = FunctionFactory.available_functions()


def create_graph_card(header_id: str, header_val: str, graph_id: str, graph_height: int, body_height: int) -> dbc.Card:
    # Create header content based on card type
    if header_id == "view-2d-header":
        header_content = dbc.Row(
            [
                dbc.Col(header_val, width=2),  # Reduced width to make space for button
                dbc.Col(
                    dcc.Slider(
                        id="view-2d-slider",
                        min=0,
                        max=100,
                        step=1,
                        value=10,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    width=8,  # Reduced width to make space for button
                ),
                dbc.Col(
                    dbc.Button(
                        "Add Point",
                        id="add-point-button",
                        color="primary",
                        size="sm",
                        className="w-100",
                    ),
                    width=2,
                ),
            ]
        )
    else:
        header_content = header_val

    card_content = [
        dbc.CardHeader(header_content, id=header_id),
        dbc.CardBody(
            [dcc.Graph(id=graph_id, className="h-100")],
            style={"height": graph_height},
        ),
    ]
    return dbc.Card(card_content, body=True, style={"height": body_height})


def create_title() -> html.H1:
    return html.H2("Gradient Descent Analysis", className="text-left")


def create_interactive_board() -> list[dbc.Col]:
    board_elements: list[dbc.Col] = [
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
                id="start-point-dropdown",
                options=[],  # Will be populated by callback
                value=None,
                placeholder="Select start point",
            ),
            width=2,
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
                value=12,
                step=1,
                placeholder="Number of contours",
            ),
            width=2,
        ),
        # Add new Armijo checkbox with improved alignment
        dbc.Col(
            dbc.Form(
                dbc.Checkbox(
                    id="use-armijo-checkbox",
                    label="Use Armijo Line Search",
                    value=False,
                    className="pt-2",  # Add padding top to match input height
                ),
                className="d-flex align-items-center h-100",  # Center vertically
            ),
            width=2,
        ),
    ]

    return board_elements


def create_first_row_graphs() -> list[dbc.Col]:
    first_row_graphs: list[dbc.Col] = [
        dbc.Col(
            create_graph_card("view-3d-header", "3D View", "view-3d-graph", 800, 900),
            width=6,
        ),
        dbc.Col(
            create_graph_card("top-view-header", "Top View", "top-view-graph", 800, 900),
            width=6,
        ),
    ]
    return first_row_graphs


def create_second_row_graphs() -> list[dbc.Col]:
    second_row_graphs: list[dbc.Col] = [
        dbc.Col(
            create_graph_card("view-2d-header", "Gradient Descent:", "view-2d-graph", 800, 900),
            width=6,
        ),
        dbc.Col(
            create_graph_card("result-view-header", "Optimization Path", "result-view-graph", 800, 900),
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
