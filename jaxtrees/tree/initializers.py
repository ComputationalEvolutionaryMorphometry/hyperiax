from . import JaxTree
from jax.random import PRNGKey, split
import jax
import jax.numpy as jnp
import copy

def initialize_noise(tree : JaxTree, key : PRNGKey, shape) -> JaxTree:
    new_tree = copy.deepcopy(tree)

    initialize_noise_inplace(new_tree, key, shape)

    return new_tree

def initialize_noise_inplace(tree : JaxTree, key : PRNGKey, shape) -> JaxTree:
    for node in tree.iter_bfs():
        subkey, key = split(key)
        node.data['noise'] = jax.random.normal(subkey, shape)

    return tree

def initialize_noise_leaves(tree : JaxTree, key : PRNGKey, shape) -> JaxTree:
    new_tree = copy.deepcopy(tree)

    initialize_noise_leaves_inplace(new_tree, key, shape)

    return new_tree

def initialize_noise_leaves_inplace(tree : JaxTree, key : PRNGKey, shape) -> JaxTree:
    for leaf in tree.iter_leaves():
        subkey, key = split(key)
        leaf.data['noise'] = jax.random.normal(subkey, shape)

    return tree

    
