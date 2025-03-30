from collections import OrderedDict
import dash
from dash import html, dash_table
from dash.dependencies import Input, Output, State
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from typing import Any, Dict, List, Tuple

from skimage import measure  # You may need to install scikit-image

from utils import (
    get_function_instance,
    find_grid_intersection,
    step_point_color,
    step_point_border_color,
    step_point_size,
    step_point_name,
)
from factory import FunctionFactory


def update_figures_impl(
    function_dropdown_value: str,
    epic_all_or_single_object_view: str,
    num_contours: int,
    selected_start_point_idx: int,
    slider_value: int,
    n_clicks: int,
    is_new_click: bool,  # New parameter to indicate fresh click
) -> Any:
    """Implements the figure update logic for the optimization function visualization.

    Creates and updates all four views of the optimization function based on the selected
    function and view mode.

    Args:
        function_dropdown_value: Name of the selected optimization function
        epic_all_or_single_object_view: View mode ("all" or "selected")
        num_contours: Number of contour lines to display
        selected_start_point_idx: Index of selected start point
        n_clicks: Number of button clicks
        is_new_click: True if this update was triggered by a new button click

    Returns:
        tuple: Contains eight elements in the following order:
            - header_3d_view (str): Title for 3D view
            - fig_3d_view (go.Figure): 3D surface plot of the function
            - header_top_view (str): Title for top view
            - fig_top_view (go.Figure): Contour plot from top
            - fig_2d_view (go.Figure): Side view visualization
            - header_result_view (str): Title for result view
            - fig_result_view (go.Figure): Results visualization
    """
    # Get function instance and selected start point
    function = get_function_instance(function_dropdown_value)
    start_point = function.start_points[selected_start_point_idx]

    if is_new_click:
        print(f"New button click detected in impl! Count: {n_clicks}")
        # Add your button click handling logic here

    print("Updating figures with function:", function_dropdown_value)
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

    # Calculate z-value of start point
    start_z = function.implementation(start_point[0], start_point[1])

    # Create 3D surface plot
    fig_3d_view = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale="viridis",
                showscale=False,  # Hide colorbar
                contours=dict(
                    z=dict(
                        show=True,
                        start=z_min,
                        end=z_max,
                        size=(z_max - z_min) / num_contours,
                        project=dict(z=True),  # Project onto surface
                        color="black",
                        width=2,
                        usecolormap=False,
                    ),
                    x=dict(show=False),
                    y=dict(show=False),
                ),
                showlegend=False,
            ),
        ]
    )

    # Update 3D layout with rotated camera
    fig_3d_view.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-1.5, y=-1.5, z=1.5),  # Changed from (1.5, 1.5, 1.5)
            ),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    # Add start point to 3D view
    fig_3d_view.add_trace(
        go.Scatter3d(
            x=[start_point[0]],
            y=[start_point[1]],
            z=[start_z],
            mode="markers",
            marker=dict(size=10, color="red", symbol="circle"),
            name="Start",
            showlegend=True,
        )
    )

    # Extract contours and add as Scatter3d lines
    contours = measure.find_contours(Z, level=start_z)
    for contour in contours:
        # Map contour indices to x and y coordinates
        x_contour = np.interp(contour[:, 1], [0, Z.shape[1] - 1], [x[0], x[-1]])
        y_contour = np.interp(contour[:, 0], [0, Z.shape[0] - 1], [y[0], y[-1]])
        z_contour = np.full_like(x_contour, start_z)

        # Add 3D contour line
        fig_3d_view.add_trace(
            go.Scatter3d(
                x=x_contour,
                y=y_contour,
                z=z_contour,
                mode="lines",
                line=dict(width=4, color="red"),
                name="Contour",
                showlegend=False,
            )
        )

        # Add projection to xy-plane
        fig_3d_view.add_trace(
            go.Scatter3d(
                x=x_contour,
                y=y_contour,
                z=np.full_like(z_contour, z_min),  # Project to bottom
                mode="lines",
                line=dict(
                    width=2,
                    color="red",
                ),
                opacity=0.5,  # Make projection semi-transparent
                name="Projection",
                showlegend=False,
            )
        )

    # Calculate gradient at start point
    gradient = function.gradient(start_point[0], start_point[1])
    gradient_scale = -1.0 / np.linalg.norm(
        gradient
    )  # Scale factor for gradient vector visualization
    end_point = (
        start_point[0] + gradient_scale * gradient[0],
        start_point[1] + gradient_scale * gradient[1],
    )
    end_z = z_min

    # Create a scatter trace that draws the arrow as a line in the xy-plane at z=plane_z
    x0 = start_point[0]
    y0 = start_point[1]
    dx0 = end_point[0] - x0
    dy0 = end_point[1] - y0
    arrow_trace = go.Scatter3d(
        x=[x0, end_point[0]],
        y=[y0, end_point[1]],
        z=[end_z, end_z],
        mode="lines",
        line=dict(color="red", width=3),
        marker=dict(size=5, color="red"),
        name="- grad f",  # LaTeX formula
        text="- Gradient Direction",
    )

    # Add the gradient arrow trace
    fig_3d_view.add_trace(arrow_trace)

    # Calculate intersection point with grid boundary
    gradient_direction = (gradient_scale * gradient[0], gradient_scale * gradient[1])
    intersection_point = find_grid_intersection(
        start_point, gradient_direction, (x_min, x_max), (y_min, y_max)
    )
    descent_direction = (
        intersection_point[0] - start_point[0],
        intersection_point[1] - start_point[1],
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
                colorbar=dict(title="f(x,y)"),
                contours=dict(
                    start=z_min,
                    end=z_max,
                    size=(z_max - z_min) / num_contours,
                    # Add specific contour level
                    value=start_z,
                ),
                line=dict(color="black", width=2),
            ),
            # Add highlighted contour at start point z-value
            go.Contour(
                x=x,
                y=y,
                z=Z,
                contours=dict(start=start_z, end=start_z, coloring="none"),
                line=dict(color="red", width=3),
                showscale=False,
                showlegend=False,
            ),
            go.Scatter(
                x=[start_point[0]],
                y=[start_point[1]],
                mode="markers",
                marker=dict(size=10, color="red", symbol="circle"),
                name="Start Point",
                showlegend=False,
            ),
        ]
    )

    # add gradient arrow to top view
    gradient_quiver_top_view = ff.create_quiver(
        [x0],
        [y0],
        [dx0],
        [dy0],
        scale=1.0,
        arrow_scale=0.1,
        name="- Gradient",
        line=dict(width=2, color="red"),
        hoverinfo="text+name",
        showlegend=False,
    )
    fig_top_view.add_trace(gradient_quiver_top_view.data[0])

    # Update 2D layout
    fig_top_view.update_layout(
        xaxis_title="X",
        yaxis_title="Y",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(scaleanchor="y", scaleratio=1),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    # Create line plot along gradient direction
    num_points = 100
    t = np.linspace(0, 1, num_points)
    line_x = start_point[0] + t * (intersection_point[0] - start_point[0])
    line_y = start_point[1] + t * (intersection_point[1] - start_point[1])
    line_z = np.array([function.implementation(x, y) for x, y in zip(line_x, line_y)])

    # Calculate new point position based on slider value
    step_size = slider_value / 100.0
    new_point = (
        start_point[0] + step_size * descent_direction[0],
        start_point[1] + step_size * descent_direction[1],
    )
    new_point_z = function.implementation(new_point[0], new_point[1])

    # Add gradient direction curve to 3D view
    fig_3d_view.add_trace(
        go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode="lines",
            line=dict(color="red", width=4),
            name="Path along gradient",
            showlegend=False,
        )
    )

    # Add step point to 3D view
    fig_3d_view.add_trace(
        go.Scatter3d(
            x=[new_point[0]],
            y=[new_point[1]],
            z=[new_point_z],
            mode="markers",
            marker=dict(
                color=step_point_color,
                size=step_point_size,
                line=dict(color=step_point_border_color, width=2),
            ),
            name=step_point_name,
            showlegend=True,
        )
    )

    # Add step point to top view
    fig_top_view.add_trace(
        go.Scatter(
            x=[new_point[0]],
            y=[new_point[1]],
            mode="markers",
            marker=dict(
                color=step_point_color,
                size=step_point_size,
                line=dict(color=step_point_border_color, width=2),
            ),
            name=step_point_name,
            showlegend=False,  # Hide legend in top view
        )
    )

    # Create 2D line plot showing function values along gradient direction
    # Calculate distance of new point from start point
    new_point_distance = np.sqrt(
        (new_point[0] - start_point[0]) ** 2 + (new_point[1] - start_point[1]) ** 2
    )

    # Create 2D line plot showing function values along gradient direction
    fig_2d_view = go.Figure(
        data=[
            go.Scatter(
                x=np.sqrt((line_x - start_point[0]) ** 2 + (line_y - start_point[1]) ** 2),
                y=line_z,
                mode="lines",
                line=dict(width=2, color="red"),
                name="Function along gradient",
                showlegend=True,
            ),
            go.Scatter(
                x=[0],  # Start point at distance 0
                y=[start_z],
                mode="markers",
                marker=dict(color="red", size=10),
                name="Start point",
            ),
            # Add new point based on slider
            go.Scatter(
                x=[new_point_distance],
                y=[new_point_z],
                mode="markers",
                marker=dict(
                    color=step_point_color,
                    size=step_point_size,
                    line=dict(color=step_point_border_color, width=2),
                ),
                name=step_point_name,
                showlegend=True,
            ),
        ],
        layout=dict(
            xaxis_title="Distance from start point",
            yaxis_title="Function value",
            showlegend=True,
            margin=dict(l=0, r=0, t=30, b=0),
        ),
    )

    # Add gradient direction arrow using quiver
    gradient_quiver = ff.create_quiver(
        [0],  # x starting point
        [0],  # y starting point
        [1.0],  # x direction (fixed length)
        [0],  # y direction (horizontal arrow)
        scale=1.0,
        arrow_scale=0.05,
        name="- Gradient",
        line=dict(width=2, color="red"),
        hoverinfo="text+name",
        showlegend=False,
    )
    fig_2d_view.add_trace(gradient_quiver.data[0])

    header_result_view = "Result View"
    fig_result_view = go.Figure()

    return (
        header_3d_view,
        fig_3d_view,
        header_top_view,
        fig_top_view,
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
            Output("view-2d-graph", "figure"),
            Output("result-view-header", "children"),
            Output("result-view-graph", "figure"),
        ],
        [
            Input("function-dropdown", "value"),
            Input("epic-all-or-single-object-view", "value"),
            Input("num-contours-input", "value"),
            Input("start-point-dropdown", "value"),
            Input("view-2d-slider", "value"),
            Input("add-point-button", "n_clicks"),  # Add button clicks as input
        ],
    )
    def update_figures(
        function_dropdown_value: str,
        epic_all_or_single_object_view: str,
        num_contours: int,
        selected_start_point_idx: int,
        slider_value: int,
        n_clicks: int,
    ):
        # Check if update was triggered by button click
        ctx = dash.callback_context
        is_new_click = False
        if ctx.triggered and ctx.triggered[0]["prop_id"] == "add-point-button.n_clicks":
            is_new_click = True

        return update_figures_impl(
            function_dropdown_value,
            epic_all_or_single_object_view,
            num_contours,
            selected_start_point_idx or 0,  # default to first point if None
            slider_value,
            n_clicks or 0,
            is_new_click,
        )

    @app.callback(
        Output("start-point-dropdown", "options"),
        Output("start-point-dropdown", "value"),
        Input("function-dropdown", "value"),
    )
    def update_start_points(function_name: str):
        """Update start point dropdown options based on selected function."""
        if not function_name:
            return [], None

        start_points = FunctionFactory.get_start_points(function_name)[function_name]
        options = [
            {"label": f"({x:.2f}, {y:.2f})", "value": i} for i, (x, y) in enumerate(start_points)
        ]

        return options, 0  # Select first point by default

    @app.callback(
        Output("add-point-button", "n_clicks"),
        Input("add-point-button", "n_clicks"),
    )
    def handle_add_point_click(n_clicks):
        if n_clicks is not None:
            print(f"Add Point button was clicked! Click count: {n_clicks}")
        return n_clicks
