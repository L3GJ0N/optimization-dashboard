from typing import List, Optional


from gradient_descent.utils.type_hints import Int, Float, Point2D, Range

from gradient_descent.utils.factory import FunctionFactory
from gradient_descent.optimization.optimization_functions import ExampleFunctions

_current_function: Optional[ExampleFunctions] = None


def get_function_instance(function_name: str) -> ExampleFunctions:
    """Get or create an instance of the selected optimization function.

    Args:
        function_name: Name of the function from the dropdown

    Returns:
        Instance of the selected optimization function
    """
    global _current_function

    # Create new instance if none exists or if function type changed
    if _current_function is None or function_name != _current_function.__class__.__name__:
        _current_function = FunctionFactory.create(function_name)

    return _current_function


def find_grid_intersection(
    start: Point2D,
    direction: Point2D,
    x_range: Range,
    y_range: Range,
) -> Point2D:
    """Calculate intersection of line with grid boundary.

    Args:
        start: Starting point (x, y)
        direction: Direction vector (dx, dy)
        x_range: Grid x limits (min, max)
        y_range: Grid y limits (min, max)

    Returns:
        Intersection point (x, y)
    """
    # Parametric line equation: P = start + t * direction
    # Find intersection parameters for all boundaries
    tx_min: Float = (x_range[0] - start[0]) / direction[0] if direction[0] != 0 else float("inf")
    tx_max: Float = (x_range[1] - start[0]) / direction[0] if direction[0] != 0 else float("inf")
    ty_min: Float = (y_range[0] - start[1]) / direction[1] if direction[1] != 0 else float("inf")
    ty_max: Float = (y_range[1] - start[1]) / direction[1] if direction[1] != 0 else float("inf")

    # Get all positive intersection parameters
    t_values: List[Float] = [t for t in [tx_min, tx_max, ty_min, ty_max] if t > 0]

    # Use smallest positive t (first intersection)
    if not t_values:
        return start  # Fallback if no intersection found
    t: Float = min(t_values)

    # Calculate intersection point
    return (start[0] + t * direction[0], start[1] + t * direction[1])


# Define consistent styling for step point
step_point_color: str = "rgb(0, 255, 255)"  # Cyan/bright blue
step_point_border_color: str = "rgb(0, 191, 255)"  # Slightly darker blue
step_point_size: Int = 10
step_point_name: str = "Next"

armijo_point_color: str = "pink"  # Red
armijo_point_border_color: str = "red"  # Red
armijo_point_size: Int = 10
