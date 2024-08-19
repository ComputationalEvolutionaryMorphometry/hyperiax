from jax import numpy as jnp
from jax.random import PRNGKey
from pytest import fixture
from hyperiax.tree.topology import symmetric_topology
from hyperiax.tree import HypTree
import jax




@fixture
def small_tree():
    return HypTree(symmetric_topology(5,2))


@fixture
def noise_tree():
    key = PRNGKey(0)
    t = HypTree(symmetric_topology(5,2))
    t.add_property('noise', (2,))
    t.data['noise'] = jax.random.normal(key, shape=t.data['noise'].shape)
    return t


