from dataclasses import dataclass

from gradient_descent.optimization.optimization_functions import ExampleFunctions
from gradient_descent.utils.type_hints import Float, Point2D


@dataclass
class GradientDescentResult:
    """Contains results of gradient descent optimization."""

    final_point: Point2D
    final_value: Float
    path: list[Point2D]
    f_values: list[Float]


def gradient_descent_with_line_search(
    function: ExampleFunctions,
    start_point: Point2D,
    tol: Float = 1e-5,
    max_iter: int = 50,
    max_line_search_attempts: int = 50,
) -> GradientDescentResult:
    """
    Gradient Descent with Line Search for 2D optimization problems.

    Args:
        function: Instance of ExampleFunctions containing implementation and gradient
        start_point: Initial point as (x, y) tuple
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        max_line_search_attempts: Maximum attempts for line search

    Returns:
        GradientDescentResult containing:
        - Final point as (x, y) tuple
        - Function value at final point
        - List of all intermediate points [(x0,y0), (x1,y1), ...]
        - List of function values at each point [f(x0,y0), f(x1,y1), ...]
    """
    current_point = start_point
    path: list[Point2D] = [current_point]
    f_values: list[Float] = [function.implementation(start_point[0], start_point[1])]

    for _i in range(max_iter):
        # Get gradient at current point
        grad: Point2D = function.gradient(current_point[0], current_point[1])
        grad_norm_sqr: Float = grad[0] ** 2 + grad[1] ** 2

        # Check convergence
        if grad_norm_sqr < tol**2:
            break

        # Line search to find optimal step size
        alpha: Float = 1.0
        line_search_attempts = 0
        while True:
            # Calculate potential new point
            new_point: Point2D = (
                current_point[0] - alpha * grad[0],
                current_point[1] - alpha * grad[1],
            )

            # Calculate function values
            f_current: Float = function.implementation(current_point[0], current_point[1])
            f_new: Float = function.implementation(new_point[0], new_point[1])

            # Armijo condition
            if f_new <= f_current - 0.5 * alpha * grad_norm_sqr:
                break

            alpha *= 0.5
            line_search_attempts += 1
            if line_search_attempts > max_line_search_attempts:
                break

        # Update current point and add to path
        current_point = new_point
        path.append(current_point)
        f_values.append(f_new)

    # Calculate final function value
    final_value: Float = function.implementation(current_point[0], current_point[1])

    return GradientDescentResult(final_point=current_point, final_value=final_value, path=path, f_values=f_values)


# Example usage
if __name__ == "__main__":
    from factory import FunctionFactory

    # Get example function
    function: ExampleFunctions = FunctionFactory.create("Rosenbrock")

    # Initial point
    x0: Point2D = (0.0, 0.0)

    # Perform gradient descent
    result: GradientDescentResult = gradient_descent_with_line_search(function, x0)

    print(f"Starting point: {x0}")
    print(f"Final point: {result.final_point}")
    print(f"Final value: {result.final_value}")
    print(f"Number of steps: {len(result.path) - 1}")
    print(f"Function values: {result.f_values}")
