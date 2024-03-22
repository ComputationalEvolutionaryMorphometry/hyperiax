from hyperiax.tree.builders import symmetric_tree
from hyperiax.tree.initializers import initialize_noise_inplace
from jax.random import PRNGKey

def test_symmetric_tree():
    tree = symmetric_tree(3, 100)

    assert len(tree.root.children) == 100
    for c in tree.root.children:
        assert len(c.children) == 100

def test_noise_init():
    key = PRNGKey(0)
    tree = symmetric_tree(4, 3)
    shape = (2,4)

    noise_tree = initialize_noise_inplace(tree, key, shape)

    assert (tree.root.data['noise'] == noise_tree.root.data['noise']).all()
    assert noise_tree.root.data['noise'].shape == shape