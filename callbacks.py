from collections import OrderedDict
import dash
from dash import html, dash_table
from dash.dependencies import Input, Output, State
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Any, Dict, List, Tuple

from utils import get_function_instance


def update_figures_impl(
    function_dropdown_value: str,
    epic_all_or_single_object_view: str,
    num_contours: int,
) -> Any:
    """Implements the figure update logic for the optimization function visualization.

    Creates and updates all four views of the optimization function based on the selected
    function and view mode.

    Args:
        function_dropdown_value: Name of the selected optimization function
        epic_all_or_single_object_view: View mode ("all" or "selected")

    Returns:
        tuple: Contains eight elements in the following order:
            - header_3d_view (str): Title for 3D view
            - fig_3d_view (go.Figure): 3D surface plot of the function
            - header_top_view (str): Title for top view
            - fig_top_view (go.Figure): Contour plot from top
            - header_2d_view (str): Title for 2D view
            - fig_2d_view (go.Figure): Side view visualization
            - header_result_view (str): Title for result view
            - fig_result_view (go.Figure): Results visualization
    """
    print("Updating figures with function:", function_dropdown_value)
    print("View mode:", epic_all_or_single_object_view)
    header_3d_view = f"3D View - {function_dropdown_value}"

    # Get function instance based on dropdown selection
    function = get_function_instance(function_dropdown_value)

    # Get current grid
    (x_min, x_max), (y_min, y_max) = function.grid

    # Create meshgrid for surface plot
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)

    # Calculate Z values
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = function.implementation(X[i, j], Y[i, j])

    # Calculate Z range and create contour lines
    z_min, z_max = np.min(Z), np.max(Z)
    contour_lines = np.linspace(z_min, z_max, num_contours)

    # Create 3D surface plot
    fig_3d_view = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale="viridis",
                colorbar=dict(title="Value"),
                contours=dict(
                    z=dict(
                        show=True,
                        start=z_min,
                        end=z_max,
                        size=(z_max - z_min) / num_contours,
                        color="black",
                        width=2,
                    )
                ),
            )
        ]
    )

    # Update 3D layout
    fig_3d_view.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5),
            ),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    # Initialize other views (to be implemented)
    header_top_view = "Top View"
    fig_top_view = go.Figure()

    # Create 2D contour view
    fig_top_view = go.Figure(
        data=[
            go.Contour(
                x=x,
                y=y,
                z=Z,
                colorscale="viridis",
                colorbar=dict(title="Value"),
                contours=dict(
                    start=z_min,
                    end=z_max,
                    size=(z_max - z_min) / num_contours,
                ),
                line=dict(color="black", width=2),
            )
        ]
    )

    # Update 2D layout
    fig_top_view.update_layout(
        xaxis_title="X",
        yaxis_title="Y",
        margin=dict(l=0, r=0, t=30, b=0),
    )

    header_2d_view = "2D View"
    fig_2d_view = go.Figure()
    header_result_view = "Result View"
    fig_result_view = go.Figure()

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
    """Registers all callback functions for the Dash application.

    Sets up the interactive behavior of the dashboard by connecting
    inputs and outputs through callback functions.

    Args:
        app: The Dash application instance

    Note:
        Currently registers callbacks for:
        - Updating all visualization figures based on:
            * Selected optimization function
            * View mode selection
    """

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
            Input("num-contours-input", "value"),
        ],
    )
    def update_figures(
        function_dropdown_value: str,
        epic_all_or_single_object_view: str,
        num_contours: int,
    ):
        return update_figures_impl(
            function_dropdown_value, epic_all_or_single_object_view, num_contours
        )
