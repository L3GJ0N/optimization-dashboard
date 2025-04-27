import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from typing import List, Tuple
from type_hints import Array1D, Float
from skimage import measure

from optimization_state import OptimizationState
from gd_implementations import GradientDescentResult
from utils import (
    step_point_color,
    step_point_border_color,
    step_point_size,
    step_point_name,
    armijo_point_color,
)


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
            next_state: OptimizationState = state_history[i + 1]

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
                    marker=dict(color=armijo_point_color, size=8, symbol="circle"),
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
                    line=dict(color=armijo_point_color, width=2),
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
    contours: List[Array1D] = measure.find_contours(state.Z, level=state.current_z)
    for contour in contours:
        x_contour: Array1D = np.interp(
            contour[:, 1], [0, state.Z.shape[1] - 1], [state.x_min, state.x_max]
        )
        y_contour: Array1D = np.interp(
            contour[:, 0], [0, state.Z.shape[0] - 1], [state.y_min, state.y_max]
        )
        z_contour: Array1D = np.full_like(x_contour, state.current_z)

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
    t: Array1D = np.linspace(0, 1, 100)
    line_x: Array1D = state.current_point[0] + t * state.descent_direction[0]
    line_y: Array1D = state.current_point[1] + t * state.descent_direction[1]
    line_z: Array1D = np.array(
        [state.function.implementation(x, y) for x, y in zip(line_x, line_y)]
    )

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
        path_x: List[Float] = [s.current_point[0] for s in state_history]
        path_y: List[Float] = [s.current_point[1] for s in state_history]

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
        gd_path_x: List[Float] = [point[0] for point in gd_result.path[:-1]]
        gd_path_y: List[Float] = [point[1] for point in gd_result.path[:-1]]

        # Add path line and points (excluding last point)
        fig.add_trace(
            go.Scatter(
                x=gd_path_x,
                y=gd_path_y,
                mode="lines+markers",
                line=dict(color=armijo_point_color, width=2),
                marker=dict(
                    color=armijo_point_color,
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
    gradient_quiver: go.Figure = ff.create_quiver(
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
    t: Array1D = np.linspace(0, 1, 100)
    line_x: Array1D = state.current_point[0] + t * state.descent_direction[0]
    line_y: Array1D = state.current_point[1] + t * state.descent_direction[1]

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
    t: Array1D = np.linspace(0, 1, 100)
    line_x: Array1D = state.current_point[0] + t * state.descent_direction[0]
    line_y: Array1D = state.current_point[1] + t * state.descent_direction[1]
    line_z: Array1D = np.array(
        [state.function.implementation(x, y) for x, y in zip(line_x, line_y)]
    )

    # Calculate distances from start point
    distances: Array1D = np.sqrt(
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
        path_distances: List[Float] = [0]  # Start with 0
        path_z_values: List[Float] = [state_history[0].current_z]

        # Calculate cumulative distances and function values
        for i in range(1, len(state_history)):
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
                line=dict(color=armijo_point_color, width=2),
                marker=dict(
                    color=armijo_point_color,
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

    display_name: str = current_state.function.display_name
    view_3d_header: str = f"{current_state.function.__class__.__name__}  -  {display_name}"
    top_view_header: str = f"Top View  -  {display_name}"

    return (
        view_3d_header,
        fig_3d_view,
        top_view_header,
        fig_top_view,
        fig_2d_view,
        "Optimization Path",
        fig_result_view,
    )
