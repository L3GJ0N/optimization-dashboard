from gradient_descent.optimization.optimization_state import OptimizationState


def test_state_initialization(rosenbrock_function, origin_point, standard_grid):
    """Test that the optimization state initializes correctly."""
    # Arrange & Act
    step_size_update_info = {
        "sigma-param-input": 0.5,
        "beta-param-input": 0.5,
        "is_update_param_button_clicked": False,
    }
    # Create an instance of OptimizationState
    state = OptimizationState(
        function=rosenbrock_function,
        current_point=origin_point,
        grid=standard_grid,
        num_contours=10,
        slider_value=50,
        step_size_update_info=step_size_update_info,
    )

    # Assert
    assert state.function == rosenbrock_function
    assert state.current_point == origin_point
    assert state.grid == standard_grid
    assert state.num_contours == 10
    assert state.step_size == 0.5
    assert state.alpha == 0.5
    assert state.beta == 0.5

    # Check calculated values
    assert state.x_min == -2.0
    assert state.x_max == 2.0
    assert state.y_min == -2.0
    assert state.y_max == 2.0

    # Check shape of arrays
    assert state.X.shape == state.Y.shape
    assert state.Z.shape == state.X.shape


def test_gradient_calculation(optimization_state):
    """Test that gradient calculation works correctly."""
    # The gradient at origin for Rosenbrock is (-2, 0)
    expected_gradient = (-2.0, 0.0)

    # Check that gradient is approximately correct
    assert abs(optimization_state.gradient[0] - expected_gradient[0]) < 1e-6
    assert abs(optimization_state.gradient[1] - expected_gradient[1]) < 1e-6


def test_step_point_calculation(optimization_state):
    """Test that step point is calculated correctly based on step size."""
    # Given step_size 0.5 and gradient (-2, 0),
    # step should be in positive x direction

    # Origin + step should move in negative gradient direction (positive x)
    assert optimization_state.step_point[0] > 0
    assert abs(optimization_state.step_point[1]) < 1e-6  # y should stay close to 0
