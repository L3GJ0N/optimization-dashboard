"""Type hint definitions for use across the project."""

import numpy as np
from typing import List, Tuple, TypeVar, Annotated
from numpy.typing import NDArray

# Basic numeric type aliases
Float = np.float64
Int = np.int64

# Range type (min, max) for axis bounds
Range = Tuple[Float, Float]

# Array dimension type aliases
Array1D = NDArray[Float]  # 1-dimensional array (vector), shape: (n,)
Array2D = NDArray[Float]  # 2-dimensional array (matrix), shape: (m,n)
Array3D = NDArray[Float]  # 3-dimensional array (tensor), shape: (i,j,k)
Grid2D = Array2D  # Semantic alias for 2D grid data

# Shape-specific type aliases using Annotated
Point2D = Tuple[Float, Float]
Points2D = Annotated[NDArray[Float], "shape=(n,2)"]
Points3D = Annotated[NDArray[Float], "shape=(n,3)"]

# Grid component type aliases
XYPair = Tuple[Float, Float]
GridDef = List[XYPair]  # Usually [(x_min, x_max), (y_min, y_max)]

# Type variables for shapes
N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
