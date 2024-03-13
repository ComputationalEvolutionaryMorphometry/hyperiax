
from . import JaxTree, JaxNode
from jax.random import PRNGKey, split
import jax
import jax.numpy as jnp
import copy
from typing import Callable
from functools import partial

def update_noise_inplace(update_f: Callable[[JaxNode,float],float], tree : JaxTree, key=False, save_noise=False) -> JaxTree:
    for node in tree.iter_bfs():
        subkey, key = split(key)
        if save_noise:
            node.data['old_noise'] = node.data['noise']
        node.data['noise'] = update_f(node,jax.random.normal(subkey, node.data['noise'].shape))

    return tree