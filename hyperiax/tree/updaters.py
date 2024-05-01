from . import HypTree, TreeNode
from jax.random import split
import jax
from typing import Callable

def update_noise_inplace(update_f: Callable[[TreeNode,float],float], tree : HypTree, key: jax.random.PRNGKey=jax.random.PRNGKey(0), save_noise: bool=False) -> HypTree:
    """
    Update the noise in each node of the tree, inplace.

    :param update_f: the way to update the noise
    :param tree: The tree to update the noise in
    :param key: The key to generate the noise with, defaults to jax.random.PRNGKey(0)
    :param save_noise: Whether to save the old noise, defaults to False
    :return: The tree with updated noise
    """
    for node in tree.iter_bfs():
        subkey, key = split(key)
        if save_noise:
            node.data['old_noise'] = node.data['noise']
        node.data['noise'] = update_f(node,jax.random.normal(subkey, node.data['noise'].shape))

    return tree