import pytest

from gradient_descent.main import setup
from gradient_descent.optimization.optimization_functions import ExampleFunctions
from gradient_descent.optimization.optimization_state import OptimizationState
from gradient_descent.utils.factory import FunctionFactory
from gradient_descent.utils.type_hints import GridDef, Point2D


@pytest.fixture
def rosenbrock_function() -> ExampleFunctions:
    """Fixture providing a Rosenbrock function instance."""
    return FunctionFactory.create("Rosenbrock")


@pytest.fixture
def gaussian_function() -> ExampleFunctions:
    """Fixture providing a Gaussian function instance."""
    return FunctionFactory.create("GaussianVariation")


@pytest.fixture
def standard_grid() -> GridDef:
    """Fixture providing a standard grid definition."""
    return [(-2.0, 2.0), (-2.0, 2.0)]


@pytest.fixture
def origin_point() -> Point2D:
    """Fixture providing the origin point (0,0)."""
    return (0.0, 0.0)


@pytest.fixture
def optimization_state(rosenbrock_function, origin_point, standard_grid) -> OptimizationState:
    """Fixture providing a standard optimization state."""
    step_size_update_info = {
        "sigma-param-input": 0.5,
        "beta-param-input": 0.5,
        "is_update_param_button_clicked": False,
    }
    return OptimizationState(
        function=rosenbrock_function,
        current_point=origin_point,
        grid=standard_grid,
        num_contours=10,
        slider_value=50,
        step_size_update_info=step_size_update_info,
    )


@pytest.fixture
def dash_app():
    """Fixture providing a Dash app instance."""
    return setup()


@pytest.fixture
def test_client(dash_app):
    """Fixture providing a test client for the Dash app."""
    with dash_app.server.test_client() as client:
        yield client


def pytest_collection_modifyitems(items):
    for item in items:
        # Auto-mark based on directory
        if "test_optimization" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "test_e2e" in item.nodeid:
            item.add_marker(pytest.mark.e2e)
