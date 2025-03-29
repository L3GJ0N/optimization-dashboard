from typing import Optional, Tuple

from factory import FunctionFactory
from optimization_functions import ExampleFunctions

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
    start: Tuple[float, float],
    direction: Tuple[float, float],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
) -> Tuple[float, float]:
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
    tx_min = (x_range[0] - start[0]) / direction[0] if direction[0] != 0 else float("inf")
    tx_max = (x_range[1] - start[0]) / direction[0] if direction[0] != 0 else float("inf")
    ty_min = (y_range[0] - start[1]) / direction[1] if direction[1] != 0 else float("inf")
    ty_max = (y_range[1] - start[1]) / direction[1] if direction[1] != 0 else float("inf")

    # Get all positive intersection parameters
    t_values = [t for t in [tx_min, tx_max, ty_min, ty_max] if t > 0]

    # Use smallest positive t (first intersection)
    if not t_values:
        return start  # Fallback if no intersection found
    t = min(t_values)

    # Calculate intersection point
    return (start[0] + t * direction[0], start[1] + t * direction[1])
