class OptimizationState:
    """Holds all data needed for visualization of optimization state."""

    def __init__(self, function, start_point, grid, num_contours, slider_value):
        self.function = function
        self.current_point = start_point
        self.grid = grid
        self.num_contours = num_contours
        self.step_size = slider_value / 100.0

        # Calculate basic grid data
        self.x_min, self.x_max = grid[0]
        self.y_min, self.y_max = grid[1]
        self.x = np.linspace(self.x_min, self.x_max, 100)
        self.y = np.linspace(self.y_min, self.y_max, 100)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Calculate function values
        self.Z = self._calculate_function_values()
        self.z_min, self.z_max = np.min(self.Z), np.max(self.Z)
        self.current_z = function.implementation(start_point[0], start_point[1])

        # Calculate gradient and path information
        self.gradient = function.gradient(start_point[0], start_point[1])
        self.gradient_scale = -1.0 / np.linalg.norm(self.gradient)
        self.descent_direction = self._calculate_descent_direction()
        self.step_point = self._calculate_step_point()

    def _calculate_function_values(self):
        Z = np.zeros_like(self.X)
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                Z[i, j] = self.function.implementation(self.X[i, j], self.Y[i, j])
        return Z

    def _calculate_descent_direction(self):
        direction = (self.gradient_scale * self.gradient[0], self.gradient_scale * self.gradient[1])
        intersection = find_grid_intersection(
            self.current_point, direction, (self.x_min, self.x_max), (self.y_min, self.y_max)
        )
        return (intersection[0] - self.current_point[0], intersection[1] - self.current_point[1])

    def _calculate_step_point(self):
        return (
            self.current_point[0] + self.step_size * self.descent_direction[0],
            self.current_point[1] + self.step_size * self.descent_direction[1],
        )


def create_visualization(state: OptimizationState):
    """Creates all visualization figures based on optimization state."""
    fig_3d_view = create_3d_view(state)
    fig_top_view = create_top_view(state)
    fig_2d_view = create_2d_view(state)
    fig_result_view = go.Figure()  # Placeholder for now

    return (
        f"3D View - {state.function.__class__.__name__}",
        fig_3d_view,
        "Top View",
        fig_top_view,
        fig_2d_view,
        "Result View",
        fig_result_view,
    )


def update_figures_impl(
    function_dropdown_value: str,
    epic_all_or_single_object_view: str,
    num_contours: int,
    selected_start_point_idx: int,
    slider_value: int,
    n_clicks: int,
    is_new_click: bool,
) -> Any:
    """Main callback implementation with separated data and visualization."""
    # Get function instance and selected start point
    function = get_function_instance(function_dropdown_value)
    start_point = function.start_points[selected_start_point_idx]

    # Create state object containing all computed data
    state = OptimizationState(
        function=function,
        start_point=start_point,
        grid=function.grid,
        num_contours=num_contours,
        slider_value=slider_value,
    )

    # Generate visualizations based on state
    return create_visualization(state)
