from typing import Dict, List, Tuple, Type
from optimization_functions import ExampleFunctions, Rosebrock, GaussianVariation


class FunctionFactory:
    """Factory class for creating ExampleFunctions instances."""

    _creators: Dict[str, Type[ExampleFunctions]] = {
        "Rosebrock": Rosebrock,
        "GaussianVariation": GaussianVariation,
        # Add more functions here as they are implemented
    }

    @classmethod
    def create(cls, function_name: str) -> ExampleFunctions:
        """Creates an instance of the requested function.

        Args:
            function_name: Name of the function to create

        Returns:
            Instance of ExampleFunctions

        Raises:
            ValueError: If function_name is not registered
        """
        creator = cls._creators.get(function_name)
        if not creator:
            raise ValueError(f"Unknown function: {function_name}")
        return creator()

    @classmethod
    def register(cls, name: str, creator: Type[ExampleFunctions]) -> None:
        """Registers a new function type.

        Args:
            name: Name to register the function under
            creator: The class to instantiate for this function
        """
        cls._creators[name] = creator

    @classmethod
    def available_functions(cls) -> list[str]:
        """Returns list of all registered function names."""
        return list(cls._creators.keys())

    @classmethod
    def get_start_points(cls, function_name: str = None) -> Dict[str, List[Tuple[float, float]]]:
        """Get start points for specified function or all registered functions.

        Args:
            function_name: Optional name of specific function

        Returns:
            Dictionary mapping function names to their start points

        Raises:
            ValueError: If specified function_name is not registered
        """
        if function_name:
            if function_name not in cls._creators:
                raise ValueError(f"Unknown function: {function_name}")
            instance = cls.create(function_name)
            return {function_name: instance.start_points}

        return {name: cls.create(name).start_points for name in cls._creators.keys()}

    @classmethod
    def print_start_points(cls, function_name: str = None) -> None:
        """Print start points for specified function or all registered functions.

        Args:
            function_name: Optional name of specific function
        """
        start_points = cls.get_start_points(function_name)
        for fname, points in start_points.items():
            print(f"{fname} start points:")
            for i, point in enumerate(points, 1):
                print(f"  {i}. ({point[0]:.2f}, {point[1]:.2f})")


# Example usage
# def main():
#     # Create instances using the factory
#     rosebrock = FunctionFactory.create("Rosebrock")
#     gaussian = FunctionFactory.create("GaussianVariation")

#     # Get list of available functions
#     print(FunctionFactory.available_functions())

#     # Register a new function type (if you implement more)
#     # FunctionFactory.register("NewFunction", NewFunctionClass)

#     # Use the created instances
#     x, y = 1.0, 1.0
#     print(f"Rosebrock at ({x}, {y}): {rosebrock.implementation(x, y)}")
#     print(f"Gaussian at ({x}, {y}): {gaussian.implementation(x, y)}")
