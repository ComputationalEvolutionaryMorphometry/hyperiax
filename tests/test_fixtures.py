from jax import numpy as jnp
from jax.random import PRNGKey
from pytest import fixture

from hyperiax.execution import LevelwiseTreeExecutor
from hyperiax.models import UpDownLambda
from hyperiax.tree.builders import THeight_legacy
from hyperiax.tree.initializers import initialize_noise_leaves


@fixture
def small_tree():
    return THeight_legacy(5,2)


@fixture
def noise_tree():
    key = PRNGKey(0)
    t = THeight_legacy(1, 2)
    t = initialize_noise_leaves(t, key, (2,))
    return t


@fixture
def phony_executor():
    up = lambda noise, key, params: 2*noise
    down = lambda noise, parent, upmsg, key, params: noise.sqrt()
    fuse = lambda _,points: points.sum(0)

    model = UpDownLambda(up,fuse,down)

    exe = LevelwiseTreeExecutor(model, batch_size=20)
    return exe
