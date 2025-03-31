from abc import ABC, abstractmethod
from typing import Any, List, Tuple
import math


class ExampleFunctions(ABC):
    """Abstract base class defining interface for implementations and derivatives."""

    def __init__(self, display_name: str):
        self._path: List[Tuple[float, float]] = []
        self._display_name = display_name

    @property
    def display_name(self) -> str:
        """Get the LaTeX-style display name of the function."""
        return self._display_name

    @property
    def path(self) -> List[Tuple[float, float]]:
        """Get the current optimization path."""
        return self._path

    @path.setter
    def path(self, value: List[Tuple[float, float]]) -> None:
        """Update the optimization path.

        Args:
            value: List of (x, y) coordinate tuples
        """
        if not isinstance(value, list):
            raise TypeError("Path must be a list")
        for point in value:
            if not (
                isinstance(point, tuple)
                and len(point) == 2
                and all(isinstance(x, (int, float)) for x in point)
            ):
                raise ValueError("Path must contain tuples of 2 float values")
        self._path = value

    @property
    @abstractmethod
    def start_points(self) -> List[Tuple[float, float]]:
        """Abstract property that returns the initial points for optimization."""
        pass

    @abstractmethod
    def implementation(self, x: float, y: float) -> float:
        """Virtual method that must be implemented by derived classes."""
        pass

    @abstractmethod
    def gradient(self, x: float, y: float) -> Tuple[float, float]:
        """Virtual method that must be implemented by derived classes."""
        pass

    @property
    @abstractmethod
    def grid(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Abstract property that returns the grid ranges ((x_min, x_max), (y_min, y_max))."""
        pass

    @grid.setter
    @abstractmethod
    def grid(self, value: Tuple[Tuple[float, float], Tuple[float, float]]) -> None:
        """Abstract property setter for grid ranges."""
        pass


class Rosebrock(ExampleFunctions):
    def __init__(self):
        display_name = r"$ f(x,y) = (1-x)^2 + 100(y-x^2)^2 $"
        super().__init__(display_name)
        self._x_range = (-1.5, 1.5)  # default x range
        self._y_range = (-1.5, 1.5)  # default y range
        self._start_points = [
            (-1.0, -1.0),
            (0.0, 0.0),
            (1.0, 1.0),
        ]  # default start points

    @property
    def start_points(self) -> List[Tuple[float, float]]:
        return self._start_points

    def implementation(self, x: float, y: float) -> float:
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2  # Rosenbrock function

    def gradient(self, x: float, y: float) -> Tuple[float, float]:
        dx = -2 * (1 - x) - 400 * x * (y - x**2)
        dy = 200 * (y - x**2)
        return (dx, dy)

    @property
    def grid(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (self._x_range, self._y_range)

    @grid.setter
    def grid(self, value: Tuple[Tuple[float, float], Tuple[float, float]]) -> None:
        x_range, y_range = value
        if not (isinstance(x_range, tuple) and isinstance(y_range, tuple)):
            raise TypeError("Grid ranges must be tuples")
        if not (len(x_range) == 2 and len(y_range) == 2):
            raise ValueError("Grid ranges must contain exactly 2 values each")
        if not (x_range[0] < x_range[1] and y_range[0] < y_range[1]):
            raise ValueError("Grid range values must be in ascending order")

        self._x_range = x_range
        self._y_range = y_range


class GaussianVariation(ExampleFunctions):
    def __init__(self):
        display_name = r"$ f(x,y) = (x + \sin(y)) e^{-x^2-y^2} $"
        super().__init__(display_name)
        self._x_range = (-2.0, 2.0)  # default x range
        self._y_range = (-2.0, 2.0)  # default y range
        self._start_points = [
            (-1.5, -0.8),
            (-1.0, 1.4),
            (1.0, 0.0),
        ]  # default start points

    @property
    def start_points(self) -> List[Tuple[float, float]]:
        return self._start_points

    def implementation(self, x: float, y: float) -> float:
        return (x + math.sin(y)) * math.exp(-(x**2) - y**2)

    def gradient(self, x: float, y: float) -> Tuple[float, float]:
        exp_term = math.exp(-(x**2) - y**2)
        # dx = (1 - 2x(x + sin(y)))exp(-x² - y²)
        dx = (1 - 2 * x * (x + math.sin(y))) * exp_term
        # dy = (cos(y) - 2y(x + sin(y)))exp(-x² - y²)
        dy = (math.cos(y) - 2 * y * (x + math.sin(y))) * exp_term
        return (dx, dy)

    @property
    def grid(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (self._x_range, self._y_range)

    @grid.setter
    def grid(self, value: Tuple[Tuple[float, float], Tuple[float, float]]) -> None:
        x_range, y_range = value
        if not (isinstance(x_range, tuple) and isinstance(y_range, tuple)):
            raise TypeError("Grid ranges must be tuples")
        if not (len(x_range) == 2 and len(y_range) == 2):
            raise ValueError("Grid ranges must contain exactly 2 values each")
        if not (x_range[0] < x_range[1] and y_range[0] < y_range[1]):
            raise ValueError("Grid range values must be in ascending order")

        self._x_range = x_range
        self._y_range = y_range


# Usage example
# def main():
#     ros = Rosebrock()

#     # Set new grid ranges
#     ros.grid = ((-2.0, 2.0), (-1.0, 1.0))

#     # Get current grid
#     x_range, y_range = ros.grid
#     print(f"X range: {x_range}")  # (-2.0, 2.0)
#     print(f"Y range: {y_range}")  # (-1.0, 1.0)

#     # Calculate function value and gradient at a point
#     x, y = 1.0, 1.0
#     value = ros.implementation(x, y)
#     grad = ros.gradient(x, y)
#     print(f"f({x}, {y}) = {value}")
#     print(f"∇f({x}, {y}) = {grad}")
