from typing import Tuple, List
from optimization_functions import ExampleFunctions


def gradient_descent_with_line_search(
    function: ExampleFunctions,
    start_point: Tuple[float, float],
    tol: float = 1e-5,
    max_iter: int = 1000,
    max_line_search_attempts: int = 50,
) -> Tuple[Tuple[float, float], float, List[Tuple[float, float]]]:
    """
    Gradient Descent with Line Search for 2D optimization problems.

    Args:
        function: Instance of ExampleFunctions containing implementation and gradient
        start_point: Initial point as (x, y) tuple
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        max_line_search_attempts: Maximum attempts for line search

    Returns:
        Tuple containing:
        - Final point as (x, y) tuple
        - Function value at final point
        - List of all intermediate points [(x0,y0), (x1,y1), ...]
    """
    current_point = start_point
    path: List[Tuple[float]] = [current_point]  # Initialize path with start point

    for i in range(max_iter):
        # Get gradient at current point
        grad: Tuple[float] = function.gradient(current_point[0], current_point[1])
        grad_norm_sqr: float = grad[0] ** 2 + grad[1] ** 2

        # Check convergence
        if grad_norm_sqr < tol**2:
            break

        # Line search to find optimal step size
        alpha = 1.0
        line_search_attempts = 0
        while True:
            # Calculate potential new point
            new_point: Tuple[float] = (
                current_point[0] - alpha * grad[0],
                current_point[1] - alpha * grad[1],
            )

            # Calculate function values
            f_current: float = function.implementation(current_point[0], current_point[1])
            f_new: float = function.implementation(new_point[0], new_point[1])

            # Armijo condition
            if f_new <= f_current - 0.5 * alpha * grad_norm_sqr:
                break

            alpha *= 0.5
            line_search_attempts += 1
            if line_search_attempts > max_line_search_attempts:
                break

        # Update current point and add to path
        current_point: Tuple[float] = new_point
        path.append(current_point)

    # Calculate final function value
    final_value: float = function.implementation(current_point[0], current_point[1])

    return current_point, final_value, path


# Example usage
if __name__ == "__main__":
    from factory import FunctionFactory

    # Get example function
    function: ExampleFunctions = FunctionFactory.create("Rosebrock")

    # Initial point
    x0: Tuple[float] = (0.0, 0.0)

    # Perform gradient descent
    final_point, final_value, optimization_path = gradient_descent_with_line_search(function, x0)

    print(f"Starting point: {x0}")
    print(f"Final point: {final_point}")
    print(f"Final value: {final_value}")
    print(f"Number of steps: {len(optimization_path) - 1}")
