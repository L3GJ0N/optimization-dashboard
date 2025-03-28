from typing import Optional

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
