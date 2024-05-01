from typing import Tuple
from . import HypTree
from jax.random import split
import jax
import jax.numpy as jnp
import copy

def initialize_noise(tree : HypTree, key : jax.random.PRNGKey, shape: tuple) -> HypTree:
    """ 
    Initializes a random value of shape `shape` with key `noise` in each node,
    and returns a new tree with the noise inserted.

    :param tree: The tree to initialize the noise in
    :param key: The key to generate the noise with
    :param shape: The shape of the noise within each node
    :return: A new tree with noise inserted
    """ 
    new_tree = copy.deepcopy(tree)

    initialize_noise_inplace(new_tree, key, shape)

    return new_tree

def initialize_noise_inplace(tree : HypTree, key : jax.random.PRNGKey, shape: tuple) -> HypTree:
    """ 
    Initializes a random value of shape `shape` with key `noise` in each node, inplace.

    :param tree: The tree to initialize the noise in
    :param key: The key to generate the noise with
    :param shape: The shape of the noise within each node
    :return: The tree with noise inserted
    """    
    for node in tree.iter_bfs():
        subkey, key = split(key)
        node.data['noise'] = jax.random.normal(subkey, shape)

    return tree

def initialize_noise_leaves(tree : HypTree, key : jax.random.PRNGKey, shape: tuple) -> HypTree:
    """ 
    Initializes a random value of shape `shape` with key `noise` in each leaf,
    and returns a new tree with the noise inserted.

    :param tree: The tree to initialize the noise in
    :param key: The key to generate the noise with
    :param shape: The shape of the noise within each leaf
    :return: A new tree with noise inserted at the leaves
    """    
    new_tree = copy.deepcopy(tree)

    initialize_noise_leaves_inplace(new_tree, key, shape)

    return new_tree

def initialize_noise_leaves_inplace(tree : HypTree, key : jax.random.PRNGKey, shape: tuple) -> HypTree:
    """ 
    Initializes a random value of shape `shape` with key `noise` in each leaf, inplace.

    :param tree: The tree to initialize the noise in
    :param key: The key to generate the noise with
    :param shape: The shape of the noise within each leaf
    :return: The tree with noise inserted at the leaves
    """    
    for leaf in tree.iter_leaves():
        subkey, key = split(key)
        leaf.data['noise'] = jax.random.normal(subkey, shape)

    return tree

    
