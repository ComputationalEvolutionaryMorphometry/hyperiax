import pytest

from jax.random import key, normal
import jax.numpy as jnp

from hyperiax.tree.topology import symmetric_topology
from hyperiax.tree import HypTree
from hyperiax.mcmc.parameters import VarianceParameter, FlatParameter
from hyperiax.mcmc.parameterstore import ParameterStore


@pytest.fixture
def rng_key():
    return key(42)


@pytest.fixture
def tree(rng_key):
    tree = HypTree(symmetric_topology(3, 2))
    tree.add_property("value", shape=(2,))
    tree.add_property("noise", shape=(2,))
    tree.data["value"] = jnp.ones_like(tree.data["value"])
    tree.data["noise"] = normal(rng_key, shape=tree.data["value"].shape) * 0.1
    return tree


@pytest.fixture
def params():
    return ParameterStore(
        {
            "dummy": VarianceParameter(
                value=0.1,
                proposal="log_normal",
                proposal_dist_hparams={
                    "min": None,
                    "max": None,
                },
                proposal_var=0.1,
                prior="uniform",
                prior_dist_hparams={
                    "min": None,
                    "max": None,
                },
            ),
            "obs_var": VarianceParameter(
                value=0.1,
                proposal="log_normal",
                proposal_dist_hparams={
                    "min": 1e-5,
                    "max": None,
                },
                proposal_var=0.1,
                prior="inv_gamma",
                prior_dist_hparams={
                    "alpha": 2.0,
                    "beta": 0.003,
                },
            ),
        }
    )


@pytest.fixture
def noise(rng_key):
    return normal(rng_key, shape=(100, 2)) * 0.1


@pytest.fixture
def data(tree):
    return jnp.ones_like(tree.data["value"][tree.is_leaf])
