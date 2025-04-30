from unittest.mock import patch

from gradient_descent.ui.callbacks import (
    handle_add_point_click_logic,
    update_start_points_logic,
)


class TestUpdateStartPointsLogic:
    """Tests for update_start_points_logic function."""

    def test_empty_function_name(self):
        """Test with empty function name."""
        # Act
        options, selected_idx = update_start_points_logic("")

        # Assert
        assert options == []
        assert selected_idx is None

    def test_none_function_name(self):
        """Test with None function name."""
        # Act
        options, selected_idx = update_start_points_logic(None)

        # Assert
        assert options == []
        assert selected_idx is None

    @patch("gradient_descent.ui.callbacks.FunctionFactory.get_start_points")
    def test_valid_function_name(self, mock_get_start_points):
        """Test with a valid function name."""
        # Arrange
        function_name = "Rosenbrock"
        mock_start_points = {function_name: [(0.0, 0.0), (1.0, 1.0)]}
        mock_get_start_points.return_value = mock_start_points

        # Act
        options, selected_idx = update_start_points_logic(function_name)

        # Assert
        assert len(options) == 2
        assert options[0]["label"] == "(0.00, 0.00)"
        assert options[0]["value"] == 0
        assert options[1]["label"] == "(1.00, 1.00)"
        assert options[1]["value"] == 1
        assert selected_idx == 0  # Should select first point by default
        mock_get_start_points.assert_called_once_with(function_name)

    @patch("gradient_descent.ui.callbacks.FunctionFactory.get_start_points")
    def test_option_format(self, mock_get_start_points):
        """Test the format of returned options."""
        # Arrange
        function_name = "GaussianVariation"
        mock_start_points = {function_name: [(1.23, 4.56)]}
        mock_get_start_points.return_value = mock_start_points

        # Act
        options, selected_idx = update_start_points_logic(function_name)

        # Assert
        assert len(options) == 1
        assert options[0]["label"] == "(1.23, 4.56)"
        assert options[0]["value"] == 0


class TestHandleAddPointClickLogic:
    """Tests for handle_add_point_click_logic function."""

    def test_with_none_clicks(self):
        """Test with None clicks."""
        # Act
        result = handle_add_point_click_logic(None)

        # Assert
        assert result is None

    @patch("builtins.print")
    def test_with_valid_clicks(self, mock_print):
        """Test with a valid number of clicks."""
        # Arrange
        n_clicks = 3

        # Act
        result = handle_add_point_click_logic(n_clicks)

        # Assert
        assert result == 3
        mock_print.assert_called_once_with("Add Point button was clicked! Click count: 3")

    @patch("builtins.print")
    def test_with_zero_clicks(self, mock_print):
        """Test with zero clicks."""
        # Arrange
        n_clicks = 0

        # Act
        result = handle_add_point_click_logic(n_clicks)

        # Assert
        assert result == 0
        mock_print.assert_called_once_with("Add Point button was clicked! Click count: 0")
