"""BFFG precision policy tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperiax import Topology, Tree
from hyperiax.prebuilt.bffg import continuous_schema, init_continuous_tree


@pytest.fixture
def restore_x64_config():
    original = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", original)


def test_continuous_bffg_default_dtype_tracks_jax_float_config(restore_x64_config):
    jax.config.update("jax_enable_x64", True)
    topo = Topology.from_parents([0, 0])
    tree = Tree.empty(topo, continuous_schema(d=1, n_steps=2))
    tree = tree.set(edge_len=jnp.ones(topo.size))

    tree = init_continuous_tree(
        tree,
        leaf_obs=jnp.array([[1.0]]),
        obs_var=0.1,
        d=1,
        n_steps=2,
        root_val=jnp.array([0.0]),
    )

    assert tree["vals"].dtype == np.dtype(jnp.float64)
    assert tree["prec_v"].dtype == np.dtype(jnp.float64)
    assert tree["ptnl_v"].dtype == np.dtype(jnp.float64)
    assert tree["anchor"].dtype == np.dtype(jnp.float64)
    assert tree["anchor_pa"].dtype == np.dtype(jnp.float64)
