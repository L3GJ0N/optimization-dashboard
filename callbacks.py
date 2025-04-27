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
    amirjio_point_color,
)
from factory import FunctionFactory


def create_3d_view(
    state: OptimizationState,
    state_history: List[OptimizationState] = None,
    gd_result: GradientDescentResult = None,
) -> go.Figure:
    """Creates 3D surface plot with contours, points and gradient visualization.

    Args:
        state: Current optimization state
        state_history: Optional list of previous optimization states
        gd_result: Optional gradient descent result with path information
    """
    # Create base surface plot
    fig = go.Figure(
        data=[
            go.Surface(
                x=state.X,
                y=state.Y,
                z=state.Z,
                colorscale="viridis",
                showscale=False,
                contours=dict(
                    z=dict(
                        show=True,
                        start=state.z_min,
                        end=state.z_max,
                        size=(state.z_max - state.z_min) / state.num_contours,
                        project=dict(z=True),
                        color="black",
                        width=2,
                        usecolormap=False,
                    ),
                    x=dict(show=False),
                    y=dict(show=False),
                ),
                showlegend=False,
            )
        ]
    )

    # Add path points and connections from history
    if state_history:
        for i, prev_state in enumerate(state_history[:-1]):  # All but current state
            next_state = state_history[i + 1]

            # Add point
            fig.add_trace(
                go.Scatter3d(
                    x=[prev_state.current_point[0]],
                    y=[prev_state.current_point[1]],
                    z=[prev_state.current_z],
                    mode="markers",
                    marker=dict(color="gray", size=8),
                    name=f"Step {i+1}",
                    showlegend=True,
                )
            )

            # Add line to next point
            fig.add_trace(
                go.Scatter3d(
                    x=[prev_state.current_point[0], next_state.current_point[0]],
                    y=[prev_state.current_point[1], next_state.current_point[1]],
                    z=[prev_state.current_z, next_state.current_z],
                    mode="lines",
                    line=dict(color="gray", width=2),
                    showlegend=False,
                )
            )

    # Add gradient descent result path if available
    if gd_result is not None:
        # Add all points except the last one
        for i, point in enumerate(gd_result.path[:-1]):  # All but last point
            next_point = gd_result.path[i + 1]
            z = gd_result.f_values[i]
            next_z = gd_result.f_values[i + 1]

            # Add point and line as before
            fig.add_trace(
                go.Scatter3d(
                    x=[point[0]],
                    y=[point[1]],
                    z=[z],
                    mode="markers",
                    marker=dict(color=amirjio_point_color, size=8, symbol="circle"),
                    name=f"GD Step {i+1}",
                    showlegend=True,
                )
            )

            # Add line to next point
            fig.add_trace(
                go.Scatter3d(
                    x=[point[0], next_point[0]],
                    y=[point[1], next_point[1]],
                    z=[z, next_z],
                    mode="lines",
                    line=dict(color=amirjio_point_color, width=2),
                    showlegend=False,
                )
            )

        # Add final point with different styling
        final_point = gd_result.path[-1]
        final_z = gd_result.f_values[-1]

        fig.add_trace(
            go.Scatter3d(
                x=[final_point[0]],
                y=[final_point[1]],
                z=[final_z],
                mode="markers",
                marker=dict(
                    color="green",  # Different color for minimum
                    size=12,  # Slightly larger
                    symbol="circle",
                ),
                name="GD minimum",
                showlegend=True,
            )
        )

    # Add contour at current z-level and its projection
    contours = measure.find_contours(state.Z, level=state.current_z)
    for contour in contours:
        x_contour = np.interp(contour[:, 1], [0, state.Z.shape[1] - 1], [state.x_min, state.x_max])
        y_contour = np.interp(contour[:, 0], [0, state.Z.shape[0] - 1], [state.y_min, state.y_max])
        z_contour = np.full_like(x_contour, state.current_z)

        # Add contour line
        fig.add_trace(
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

        # Add projection
        fig.add_trace(
            go.Scatter3d(
                x=x_contour,
                y=y_contour,
                z=np.full_like(z_contour, state.z_min),
                mode="lines",
                line=dict(width=2, color="red"),
                opacity=0.5,
                name="Projection",
                showlegend=False,
            )
        )

    # Add current point
    fig.add_trace(
        go.Scatter3d(
            x=[state.current_point[0]],
            y=[state.current_point[1]],
            z=[state.current_z],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Current",
            showlegend=True,
        )
    )

    # Add step point
    fig.add_trace(
        go.Scatter3d(
            x=[state.step_point[0]],
            y=[state.step_point[1]],
            z=[state.function.implementation(state.step_point[0], state.step_point[1])],
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

    # Add function values along gradient direction
    t = np.linspace(0, 1, 100)
    line_x = state.current_point[0] + t * state.descent_direction[0]
    line_y = state.current_point[1] + t * state.descent_direction[1]
    line_z = np.array([state.function.implementation(x, y) for x, y in zip(line_x, line_y)])

    fig.add_trace(
        go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode="lines",
            line=dict(width=3, color="red"),
            name="Function along gradient",
            showlegend=False,  # Already shown in 2D view
        )
    )

    # Update camera view
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-1.5, y=-1.5, z=1.5),
            ),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return fig


def create_top_view(
    state: OptimizationState,
    state_history: List[OptimizationState] = None,
    gd_result: GradientDescentResult = None,
) -> go.Figure:
    """Creates top view contour plot with current point and gradient."""
    fig = go.Figure(
        data=[
            # Base contour plot
            go.Contour(
                x=state.x,
                y=state.y,
                z=state.Z,
                colorscale="viridis",
                colorbar=dict(title="f(x,y)"),
                contours=dict(
                    start=state.z_min,
                    end=state.z_max,
                    size=(state.z_max - state.z_min) / state.num_contours,
                ),
                line=dict(color="black", width=2),
            ),
            # Highlight current level contour
            go.Contour(
                x=state.x,
                y=state.y,
                z=state.Z,
                contours=dict(start=state.current_z, end=state.current_z, coloring="none"),
                line=dict(color="red", width=3),
                showscale=False,
                showlegend=False,
            ),
        ]
    )

    # Add path points and connections from history
    if len(state_history) > 1:
        # Extract path coordinates
        path_x = [s.current_point[0] for s in state_history]
        path_y = [s.current_point[1] for s in state_history]

        # Add path line and points
        fig.add_trace(
            go.Scatter(
                x=path_x,
                y=path_y,
                mode="lines+markers",
                line=dict(color="gray", width=2),
                marker=dict(
                    color="gray",
                    size=10,
                    symbol="circle",
                ),
                name="Optimization path",
                showlegend=False,
            )
        )

        # Add step labels
        for i, prev_state in enumerate(state_history[:-1]):
            fig.add_annotation(
                x=prev_state.current_point[0],
                y=prev_state.current_point[1],
                text=f"{i+1}",
                showarrow=False,
                font=dict(size=10, color="white"),
            )

    # Add gradient descent result path if available
    if gd_result is not None:
        # Extract path coordinates from gradient descent result (excluding last point)
        gd_path_x = [point[0] for point in gd_result.path[:-1]]
        gd_path_y = [point[1] for point in gd_result.path[:-1]]

        # Add path line and points (excluding last point)
        fig.add_trace(
            go.Scatter(
                x=gd_path_x,
                y=gd_path_y,
                mode="lines+markers",
                line=dict(color=amirjio_point_color, width=2),
                marker=dict(
                    color=amirjio_point_color,
                    size=10,
                    symbol="circle",
                ),
                name="Gradient Descent Path",
                showlegend=True,
            )
        )

        # Add final point separately with different styling
        final_point = gd_result.path[-1]
        fig.add_trace(
            go.Scatter(
                x=[final_point[0]],
                y=[final_point[1]],
                mode="markers",
                marker=dict(
                    color="green",
                    size=12,
                    symbol="circle",
                ),
                name="GD minimum",
                showlegend=True,
            )
        )

    # Add current point
    fig.add_trace(
        go.Scatter(
            x=[state.current_point[0]],
            y=[state.current_point[1]],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Current",
            showlegend=False,
        )
    )

    # Add step point
    fig.add_trace(
        go.Scatter(
            x=[state.step_point[0]],
            y=[state.step_point[1]],
            mode="markers",
            marker=dict(
                color=step_point_color,
                size=step_point_size,
                line=dict(color=step_point_border_color, width=2),
            ),
            name=step_point_name,
            showlegend=False,  # Already shown in 3D view
        )
    )

    # Add gradient arrow using quiver
    gradient_quiver = ff.create_quiver(
        [state.current_point[0]],
        [state.current_point[1]],
        [-state.normalized_gradient[0]],  # Negative gradient for descent
        [-state.normalized_gradient[1]],
        scale=1.0,
        arrow_scale=0.1,
        name="- Gradient",
        line=dict(width=2, color="red"),
        showlegend=False,
    )
    fig.add_trace(gradient_quiver.data[0])

    # Add line along gradient direction
    t = np.linspace(0, 1, 100)
    line_x = state.current_point[0] + t * state.descent_direction[0]
    line_y = state.current_point[1] + t * state.descent_direction[1]

    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line=dict(width=1, color="red", dash="dash"),
            name="Path along gradient",
            showlegend=False,  # Already shown in 3D view
        )
    )

    # Update layout
    fig.update_layout(
        xaxis_title="X",
        yaxis_title="Y",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(scaleanchor="y", scaleratio=1),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return fig


def create_2d_view(
    state: OptimizationState,
    state_history: List[OptimizationState] = None,
    gd_result: GradientDescentResult = None,
) -> go.Figure:
    """Creates 2D line plot showing function values along gradient direction."""
    # Calculate points along gradient direction
    t = np.linspace(0, 1, 100)
    line_x = state.current_point[0] + t * state.descent_direction[0]
    line_y = state.current_point[1] + t * state.descent_direction[1]
    line_z = np.array([state.function.implementation(x, y) for x, y in zip(line_x, line_y)])

    # Calculate distances from start point
    distances = np.sqrt(
        (line_x - state.current_point[0]) ** 2 + (line_y - state.current_point[1]) ** 2
    )

    fig = go.Figure(
        data=[
            # Function along gradient
            go.Scatter(
                x=distances,
                y=line_z,
                mode="lines",
                line=dict(width=2, color="red"),
                name="f along grad",
            ),
            # Current point
            go.Scatter(
                x=[0],
                y=[state.current_z],
                mode="markers",
                marker=dict(color="red", size=10),
                name="Current",
            ),
            # Step point if applicable
            go.Scatter(
                x=[np.linalg.norm(state.descent_direction) * state.step_size],
                y=[state.function.implementation(state.step_point[0], state.step_point[1])],
                mode="markers",
                marker=dict(
                    color=step_point_color,
                    size=10,
                    line=dict(color=step_point_border_color, width=2),
                ),
                name=step_point_name,
            ),
        ]
    )

    # Update layout
    fig.update_layout(
        xaxis_title="Distance from current point / Step number",
        yaxis_title="Function value",
        showlegend=True,
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return fig


def create_2d_loss_view(
    state: OptimizationState,
    state_history: List[OptimizationState] = None,
    gd_result: GradientDescentResult = None,
) -> go.Figure:
    fig = go.Figure()
    if state_history:
        path_distances = [0]  # Start with 0
        path_z_values = [state_history[0].current_z]

        # Calculate cumulative distances and function values
        for i in range(1, len(state_history)):
            prev = state_history[i - 1].current_point
            curr = state_history[i].current_point
            dist = np.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)
            path_distances.append(i)
            path_z_values.append(state_history[i].current_z)

        # Add path trace
        fig.add_trace(
            go.Scatter(
                x=path_distances,
                y=path_z_values,
                mode="lines+markers",
                line=dict(color="gray", width=2),
                marker=dict(
                    color="gray",
                    size=8,
                    symbol="circle",
                ),
                name="Optimization path",
            )
        )

    # Add gradient descent result if available
    if gd_result is not None:
        # Create step numbers array
        steps = list(range(len(gd_result.f_values)))

        # Add trace for gradient descent path
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=gd_result.f_values,
                mode="lines+markers",
                line=dict(color=amirjio_point_color, width=2),
                marker=dict(
                    color=amirjio_point_color,
                    size=10,
                    symbol="circle",
                ),
                name="Gradient Descent Armijo",
            )
        )

    # Update layout
    fig.update_layout(
        xaxis_title="Step number",
        yaxis_title="Function value at each step",
        showlegend=True,
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return fig


def create_visualization(
    current_state: OptimizationState,
    state_history: List[OptimizationState],
    gd_result: GradientDescentResult = None,
) -> Tuple[str | go.Figure]:
    """Creates all visualization figures based on optimization state."""
    fig_3d_view: go.Figure = create_3d_view(current_state, state_history, gd_result)
    fig_top_view: go.Figure = create_top_view(current_state, state_history, gd_result)
    fig_2d_view: go.Figure = create_2d_view(current_state, state_history, gd_result)
    fig_result_view: go.Figure = create_2d_loss_view(current_state, state_history, gd_result)

    display_name = current_state.function.display_name
    view_3d_header = f"{current_state.function.__class__.__name__}  -  {display_name}"
    top_view_header = f"Top View  -  {display_name}"

    return (
        view_3d_header,
        fig_3d_view,
        top_view_header,
        fig_top_view,
        fig_2d_view,
        "Optimization Path",
        fig_result_view,
    )


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
    start_point: Tuple[float] = function.start_points[selected_start_point_idx]

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
