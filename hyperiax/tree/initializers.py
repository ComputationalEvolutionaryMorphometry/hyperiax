from . import HypTree
from jax.random import PRNGKey, split
import jax
import jax.numpy as jnp
import copy

def initialize_noise(tree : HypTree, key : PRNGKey, shape) -> HypTree:
    """Initializes a random value of shape `shape` with key `noise` in each node

    Args:
        tree (HypTree): the tree to generate in
        key (PRNGKey): the key to generate with
        shape (tuple): the shape of the noise

    Returns:
        HypTree: the tree with noise inserted
    """
    new_tree = copy.deepcopy(tree)

    initialize_noise_inplace(new_tree, key, shape)

    return new_tree

def initialize_noise_inplace(tree : HypTree, key : PRNGKey, shape) -> HypTree:
    """Initializes a random value of shape `shape` with key `noise` in each node, inplace.

    Args:
        tree (HypTree): the tree to generate in
        key (PRNGKey): the key to generate with
        shape (tuple): the shape of the noise

    Returns:
        HypTree: the tree with noise inserted
    """
    for node in tree.iter_bfs():
        subkey, key = split(key)
        node.data['noise'] = jax.random.normal(subkey, shape)

    return tree

def initialize_noise_leaves(tree : HypTree, key : PRNGKey, shape) -> HypTree:
    """Initializes a random value of shape `shape` with key `noise` in each leaf

    Args:
        tree (HypTree): the tree to generate in
        key (PRNGKey): the key to generate with
        shape (tuple): the shape of the noise

    Returns:
        HypTree: the tree with noise inserted in the leaves
    """
    new_tree = copy.deepcopy(tree)

    initialize_noise_leaves_inplace(new_tree, key, shape)

    return new_tree

def initialize_noise_leaves_inplace(tree : HypTree, key : PRNGKey, shape) -> HypTree:
    """Initializes a random value of shape `shape` with key `noise` in each leaf inplace

    Args:
        tree (HypTree): the tree to generate in
        key (PRNGKey): the key to generate with
        shape (tuple): the shape of the noise

    Returns:
        HypTree: the tree with noise inserted in the leaves
    """
    for leaf in tree.iter_leaves():
        subkey, key = split(key)
        leaf.data['noise'] = jax.random.normal(subkey, shape)

    return tree

    
