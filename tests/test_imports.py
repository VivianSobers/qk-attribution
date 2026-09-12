"""Smoke tests that the package and its dependency surface import cleanly."""


def test_package_imports():
    import qk_attribution

    assert qk_attribution.__version__


def test_circuit_tracer_available():
    from circuit_tracer.graph import Graph

    assert hasattr(Graph, "from_pt")
