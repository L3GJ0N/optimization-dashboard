import dash_core_components as dcc
from dash.development.base_component import Component

from gradient_descent.ui.layout import create_layout


def _find_by_id(root: Component, target_id: str) -> Component | None:
    """
    Recursively search a Dash component tree for a component with id=target_id.
    Returns the first matching component, or None if not found.
    """
    # Base case: does this node have an id that matches?
    if getattr(root, "id", None) == target_id:
        return root

    children: Component | list[Component] | None = getattr(root, "children", None)
    if children is None:
        return None

    if isinstance(children, list | tuple):
        for child in children:
            found = _find_by_id(child, target_id)
            if found is not None:
                return found
    else:
        return _find_by_id(children, target_id)

    return None


def _find_all_by_type(root: Component, target_type: type[Component]) -> list[Component]:
    """
    Recursively search a Dash component tree for all components
    that are instances of target_type.
    Returns a list of matching components (empty if none).
    """
    matches: list[Component] = []
    # Check current node
    if isinstance(root, target_type):
        matches.append(root)

    # Dive into children, if any
    children: Component | list[Component] | None = getattr(root, "children", None)
    if children is None:
        return matches

    if isinstance(children, list | tuple):
        for child in children:
            matches.extend(_find_all_by_type(child, target_type))
    else:
        matches.extend(_find_all_by_type(children, target_type))

    return matches


def test_layout_structure():  # noqa: C901
    """Test that the layout has all expected components."""
    # Act
    layout = create_layout()

    # Assert - check for essential components
    # Find dropdown
    dropdown = _find_by_id(layout, "function-dropdown")
    assert dropdown is not None, "Function dropdown should be present"

    # Find graphs
    graphs = _find_all_by_type(layout, dcc.Graph)
    graph_ids = {graph.id for graph in graphs}

    assert "view-3d-graph" in graph_ids, "3D view should be present"
    assert "top-view-graph" in graph_ids, "Top view should be present"

    # Check for slider
    slider = _find_by_id(layout, "view-2d-slider")
    assert slider is not None, "Step size slider should be present"
