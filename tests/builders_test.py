from jax.random import PRNGKey
from hyperiax.tree.topology import symmetric_topology

def test_symmetric_tree():
    root = symmetric_topology(3, 100)

    assert len(root.children) == 100
    for c in root.children:
        assert len(c.children) == 100