from gradient_descent.ui.visualization import create_2d_view, create_3d_view, create_top_view, create_visualization


def test_create_3d_view(optimization_state):
    """Test that 3D view is created without errors."""
    # Act
    fig = create_3d_view(optimization_state)

    # Assert
    assert fig is not None
    assert len(fig.data) >= 1  # Should have at least the surface plot
    assert "Surface" in str(type(fig.data[0]))


def test_create_top_view(optimization_state):
    """Test that top view is created without errors."""
    # Act
    fig = create_top_view(optimization_state)

    # Assert
    assert fig is not None
    assert len(fig.data) >= 1  # Should have at least the contour plot
    assert "Contour" in str(type(fig.data[0]))


def test_create_2d_view(optimization_state):
    """Test that 2D view is created without errors."""
    # Act
    fig = create_2d_view(optimization_state)

    # Assert
    assert fig is not None
    assert len(fig.data) >= 1  # Should have at least one line trace


def test_create_visualization(optimization_state):
    """Test that the complete visualization is created."""
    # Arrange
    state_history = [optimization_state]

    # Act
    results = create_visualization(optimization_state, state_history)

    # Assert
    assert len(results) == 7  # Should return 7 items (header, fig, etc)
    assert results[0] is not None  # header text
    assert results[1] is not None  # 3D figure
    assert results[3] is not None  # top view figure
    assert results[4] is not None  # 2D view
