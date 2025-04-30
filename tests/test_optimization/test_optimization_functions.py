class TestRosenbrock:
    """Test the Rosenbrock function implementation."""

    def test_function_value_at_minimum(self, rosenbrock_function):
        """Test that the function value at the known minimum (1,1) is 0."""
        # Act
        result = rosenbrock_function.implementation(1.0, 1.0)

        # Assert
        assert result == 0.0, "Rosenbrock value at (1,1) should be 0"

    def test_gradient_at_minimum(self, rosenbrock_function):
        """Test that the gradient at the minimum (1,1) is (0,0)."""
        # Act
        gradient = rosenbrock_function.gradient(1.0, 1.0)

        # Assert
        assert abs(gradient[0]) < 1e-10, "Gradient x component should be zero at minimum"
        assert abs(gradient[1]) < 1e-10, "Gradient y component should be zero at minimum"

    def test_start_points_exist(self, rosenbrock_function):
        """Test that start points are defined."""
        # Act & Assert
        assert len(rosenbrock_function.start_points) > 0, "Should have at least one start point"


class TestGaussianVariation:
    """Test the GaussianVariation function implementation."""

    def test_function_values(self, gaussian_function):
        """Test function values at specific points."""
        # Origin should have value 0
        assert gaussian_function.implementation(0.0, 0.0) == 0.0

        # Test one more point with a known value
        expected = 0.19587644987957703  # Approximate value at (0.1, 0.1)
        actual = gaussian_function.implementation(0.1, 0.1)
        assert actual > 0.0, "Gaussian value should be positive"
        assert abs(actual - expected) < 1e-2, "Gaussian value should be close to expected"
