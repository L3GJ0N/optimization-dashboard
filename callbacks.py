from collections import OrderedDict
import dash
from dash import html, dash_table
from dash.dependencies import Input, Output, State
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from typing import Any, Dict, List, Tuple

from utils import get_function_instance


def update_figures_impl(
    function_dropdown_value: str,
    epic_all_or_single_object_view: str,
) -> Any:
    print("Updating figures with function:", function_dropdown_value)
    print("View mode:", epic_all_or_single_object_view)
    header_3d_view = f"3D View - {function_dropdown_value}"
    fig_3d_view = go.Figure()
    header_top_view = "Top View"
    fig_top_view = go.Figure()
    header_2d_view = "2D View"
    fig_2d_view = go.Figure()
    header_result_view = "Result View"
    fig_result_view = go.Figure()

    # Get function instance based on dropdown selection
    function = get_function_instance(function_dropdown_value)

    # Get current grid
    x_range, y_range = function.grid
    print(f"X range: {x_range}")  # (-2.0, 2.0)
    print(f"Y range: {y_range}")  # (-1.0, 1.0)

    return (
        header_3d_view,
        fig_3d_view,
        header_top_view,
        fig_top_view,
        header_2d_view,
        fig_2d_view,
        header_result_view,
        fig_result_view,
    )


def register_all_callbacks(
    app: dash.Dash,
):
    @app.callback(
        [
            Output("view-3d-header", "children"),
            Output("view-3d-graph", "figure"),
            Output("top-view-header", "children"),
            Output("top-view-graph", "figure"),
            Output("view-2d-header", "children"),
            Output("view-2d-graph", "figure"),
            Output("result-view-header", "children"),
            Output("result-view-graph", "figure"),
        ],
        [
            Input("function-dropdown", "value"),
            Input("epic-all-or-single-object-view", "value"),
        ],
    )
    def update_figures(
        function_dropdown_value: str,
        epic_all_or_single_object_view: str,
    ):
        return update_figures_impl(function_dropdown_value, epic_all_or_single_object_view)
