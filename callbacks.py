import dash
from dash._callback import NoUpdate
from dash._callback_context import CallbackContext
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from typing import Any, Dict, List, Tuple

from skimage import measure

from optimization_state import OptimizationState
from gd_implementations import gradient_descent_with_line_search, GradientDescentResult
from optimization_functions import ExampleFunctions
from utils import (
    get_function_instance,
    step_point_color,
    step_point_border_color,
    step_point_size,
    step_point_name,
    armijo_point_color,
)
from factory import FunctionFactory
from visualization import create_visualization
from type_hints import Point2D, GridDef


def update_figures_impl(
    function_dropdown_value: str,
    epic_all_or_single_object_view: str,
    num_contours: int,
    selected_start_point_idx: int,
    slider_value: int,
    n_clicks: int,
    trigger_info: Dict[str, Any],
    use_armijo: bool,
) -> Any:
    """Main callback implementation with separated data and visualization."""
    function: ExampleFunctions = get_function_instance(function_dropdown_value)
    start_point: Point2D = function.start_points[selected_start_point_idx]

    # Initialize or reset state history
    if not hasattr(function, "state_history"):
        function.state_history = []

    # Create initial state if history is empty
    if not function.state_history:
        initial_state = OptimizationState(
            function=function,
            current_point=start_point,
            grid=function.grid,
            num_contours=num_contours,
            slider_value=slider_value,
        )
        function.state_history = [initial_state]

    # Get current state
    current_state = function.state_history[-1]

    # Reset history if start point changed
    if trigger_info["is_start_point_changed"] or trigger_info["is_function_changed"]:
        initial_state = OptimizationState(
            function=function,
            current_point=start_point,
            grid=function.grid,
            num_contours=num_contours,
            slider_value=slider_value,
        )
        function.state_history = [initial_state]
        current_state = initial_state

    # Update step size from slider
    current_state.step_size = slider_value / 100.0
    current_state.step_point = current_state._calculate_step_point()
    current_state.num_contours = num_contours

    # Add new state if button clicked
    if trigger_info["is_new_click"] and n_clicks > 0:
        print(f"Adding new state at step {len(function.state_history)}")
        new_state = OptimizationState(
            function=function,
            current_point=current_state.step_point,
            grid=function.grid,
            num_contours=num_contours,
            slider_value=slider_value,
        )
        function.state_history.append(new_state)
        current_state: OptimizationState = new_state

    # Create visualization using current state and history
    # Initialize gradient descent result
    gd_result = None

    # Check if armijo line search is used
    if use_armijo:
        # Implement Armijo line search logic here
        gd_result: GradientDescentResult = gradient_descent_with_line_search(function, start_point)
    else:
        gd_result = None

    return create_visualization(current_state, function.state_history, gd_result)


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
            Input("use-armijo-checkbox", "value"),  # Add new input
        ],
    )
    def update_figures(
        function_dropdown_value: str,
        epic_all_or_single_object_view: str,
        num_contours: int,
        selected_start_point_idx: int,
        slider_value: int,
        n_clicks: int,
        use_armijo: bool,
    ) -> NoUpdate | Any:
        # Get trigger information
        ctx: CallbackContext = dash.callback_context
        if not ctx.triggered:
            return dash.no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Determine what triggered the update
        trigger_info = {
            "is_new_click": trigger_id == "add-point-button",
            "is_function_changed": trigger_id == "function-dropdown",
            "is_start_point_changed": trigger_id == "start-point-dropdown",
            "is_slider_changed": trigger_id == "view-2d-slider",
            "is_view_mode_changed": trigger_id == "epic-all-or-single-object-view",
            "is_contours_changed": trigger_id == "num-contours-input",
            "is_armijo_changed": trigger_id == "use-armijo-checkbox",
            "trigger_id": trigger_id,
        }

        return update_figures_impl(
            function_dropdown_value,
            epic_all_or_single_object_view,
            num_contours,
            selected_start_point_idx or 0,  # default to first point if None
            slider_value,
            n_clicks or 0,
            trigger_info,
            use_armijo,
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

        start_points: List[Point2D] = FunctionFactory.get_start_points(function_name)[function_name]
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
