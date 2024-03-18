from hyperiax.tree.builders import THeight_legacy
from hyperiax.tree.initializers import initialize_noise_inplace
from jax.random import PRNGKey, split
import jax

def test_theight_legacy():
    tree = THeight_legacy(6, 2)

    leaves = list(tree.iter_leaves())

    assert len(leaves) == 2**5

def test_assign_leaves():
    tree = THeight_legacy(6, 2)
    key = PRNGKey(0)

    for leaf in tree.iter_leaves():
        subkey, key = split(key)
        leaf.data['value'] =  jax.random.normal(subkey, (2,))
    
    assert 'value' not in tree.root.data.keys()
    for l in tree.iter_leaves():
        assert l.data['value'] != None