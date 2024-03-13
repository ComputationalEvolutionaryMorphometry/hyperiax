from . import HypTree
from jax.random import PRNGKey, split
import jax
import jax.numpy as jnp
import copy

def initialize_noise(tree : HypTree, key : PRNGKey, shape) -> HypTree:
    new_tree = copy.deepcopy(tree)

    initialize_noise_inplace(new_tree, key, shape)

    return new_tree

def initialize_noise_inplace(tree : HypTree, key : PRNGKey, shape) -> HypTree:
    for node in tree.iter_bfs():
        subkey, key = split(key)
        node.data['noise'] = jax.random.normal(subkey, shape)

    return tree

def initialize_noise_leaves(tree : HypTree, key : PRNGKey, shape) -> HypTree:
    new_tree = copy.deepcopy(tree)

    initialize_noise_leaves_inplace(new_tree, key, shape)

    return new_tree

def initialize_noise_leaves_inplace(tree : HypTree, key : PRNGKey, shape) -> HypTree:
    for leaf in tree.iter_leaves():
        subkey, key = split(key)
        leaf.data['noise'] = jax.random.normal(subkey, shape)

    return tree

    
