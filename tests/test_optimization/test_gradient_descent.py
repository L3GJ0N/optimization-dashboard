from gradient_descent.optimization.gd_implementations import gradient_descent_with_line_search


def test_gradient_descent_convergence(rosenbrock_function):
    """Test that gradient descent converges to the minimum."""
    # Arrange
    start_point = (0.0, 0.0)

    # Act
    result = gradient_descent_with_line_search(rosenbrock_function, start_point, tol=1e-5, max_iter=1000)

    # Assert
    # Check that we converged to the minimum at (1,1)
    assert abs(result.final_point[0] - 1.0) < 0.01
    assert abs(result.final_point[1] - 1.0) < 0.01
    assert abs(result.final_value) < 0.01

    # Check that we have a path of points
    assert len(result.path) > 1
    assert len(result.f_values) == len(result.path)


def test_gradient_descent_early_stop(rosenbrock_function):
    """Test that gradient descent stops after max_iter."""
    # Arrange - use a very small number of iterations
    start_point = (-1.0, -1.0)
    max_iter = 5

    # Act
    result = gradient_descent_with_line_search(rosenbrock_function, start_point, max_iter=max_iter)

    # Assert
    assert len(result.path) <= max_iter + 1  # +1 for initial point
