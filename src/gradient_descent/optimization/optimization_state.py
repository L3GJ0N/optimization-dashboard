from typing import Any

import numpy as np

from gradient_descent.optimization.optimization_functions import ExampleFunctions
from gradient_descent.utils.type_hints import Array1D, Float, Grid2D, GridDef, Point2D
from gradient_descent.utils.utils import find_grid_intersection


class OptimizationState:
    """Holds all data needed for visualization of optimization state."""

    def __init__(
        self,
        function: ExampleFunctions,
        current_point: Point2D,
        grid: GridDef,
        num_contours: int,
        slider_value: int,
        step_size_update_info: dict[str, Any],
    ) -> None:
        self.function: ExampleFunctions = function
        self.current_point: Point2D = current_point
        self.grid: GridDef = grid
        self.num_contours: int = num_contours
        self.step_size: Float = slider_value / 100.0

        self.x_min: Float = grid[0][0]
        self.x_max: Float = grid[0][1]
        self.y_min: Float = grid[1][0]
        self.y_max: Float = grid[1][1]

        self.x: Array1D = np.linspace(self.x_min, self.x_max, 100)
        self.y: Array1D = np.linspace(self.y_min, self.y_max, 100)
        self.X: Grid2D = np.meshgrid(self.x, self.y)[0]
        self.Y: Grid2D = np.meshgrid(self.x, self.y)[1]

        self.alpha: Float = step_size_update_info.get("sigma-param-input", 0.5)
        self.beta: Float = step_size_update_info.get("beta-param-input", 0.5)

        # Calculate function values
        self.Z: Grid2D = self._calculate_function_values()
        self.z_min: Float = np.min(self.Z)
        self.z_max: Float = np.max(self.Z)
        self.current_z: Float = function.implementation(current_point[0], current_point[1])

        # Calculate gradient information
        self.gradient: Point2D = function.gradient(current_point[0], current_point[1])
        self.normalized_gradient: Point2D = self._calculate_normalized_gradient()
        self.gradient_scale: Float = -1.0 / np.linalg.norm(self.gradient)
        self.descent_direction: Point2D = self._calculate_descent_direction()
        self.step_point: Point2D = self._calculate_step_point()

    def _calculate_function_values(self) -> Grid2D:
        z: Grid2D = np.zeros_like(self.X)
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                z[i, j] = self.function.implementation(self.X[i, j], self.Y[i, j])
        return z

    def _calculate_normalized_gradient(self) -> Point2D:
        """Calculate normalized gradient vector."""
        norm: Float = np.linalg.norm(self.gradient)
        if norm == 0:
            return (0.0, 0.0)
        return (self.gradient[0] / norm, self.gradient[1] / norm)

    def _calculate_descent_direction(self) -> Point2D:
        direction: Point2D = (
            self.gradient_scale * self.gradient[0],
            self.gradient_scale * self.gradient[1],
        )
        intersection: Point2D = find_grid_intersection(
            self.current_point,
            direction,
            (self.x_min, self.x_max),
            (self.y_min, self.y_max),
        )
        return (
            intersection[0] - self.current_point[0],
            intersection[1] - self.current_point[1],
        )

    def _calculate_step_point(self) -> Point2D:
        return (
            self.current_point[0] + self.step_size * self.descent_direction[0],
            self.current_point[1] + self.step_size * self.descent_direction[1],
        )
