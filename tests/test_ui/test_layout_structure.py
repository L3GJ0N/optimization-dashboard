import dash_core_components as dcc

from gradient_descent.ui.layout import create_layout


def test_layout_structure():  # noqa: C901
    """Test that the layout has all expected components."""
    # Act
    layout = create_layout()

    # Assert - check for essential components
    # Find dropdown
    dropdown = None
    for component in layout.children:
        if hasattr(component, "children"):
            for child in component.children:
                if hasattr(child, "id") and child.id == "function-dropdown":
                    dropdown = child
                    break

    assert dropdown is not None, "Function dropdown should be present"

    # Find graphs
    graph_ids = []
    for component in layout.children:
        if hasattr(component, "children"):
            for child in component.children:
                if hasattr(child, "children"):
                    for grandchild in child.children:
                        if hasattr(grandchild, "id") and isinstance(grandchild, dcc.Graph):
                            graph_ids.append(grandchild.id)

    assert "view-3d" in graph_ids, "3D view should be present"
    assert "view-top" in graph_ids, "Top view should be present"

    # Check for slider
    slider = None
    for component in layout.children:
        if hasattr(component, "children"):
            for child in component.children:
                if hasattr(child, "children"):
                    for grandchild in child.children:
                        if hasattr(grandchild, "id") and grandchild.id == "slider-value":
                            slider = grandchild
                            break

    assert slider is not None, "Step size slider should be present"
