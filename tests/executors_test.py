from jaxtrees.execution import LevelwiseTreeExecutor, DependencyTreeExecutor
from .test_fixtures import small_tree, phony_executor,noise_tree
from jaxtrees.tree import JaxNode, JaxTree
import jaxtrees
from jax.random import PRNGKey
from jax import numpy as jnp
import pytest


def test_lwte_sum(noise_tree):
    assert len(noise_tree.root.children) == 2
    up = lambda noise, **kwargs: noise
    fuse = lambda messages, **kwags: {'noise' : messages.sum(0)}
    down = lambda parent_noise, **kwargs: {'noise': parent_noise/2}

    exe = LevelwiseTreeExecutor(up=up,down=down,fuse=fuse)

    manual_sum = jnp.stack([node.data['noise'] for node in noise_tree.iter_leaves()]).sum(0)

    tree_result = exe.up(noise_tree)

    assert (tree_result.root.data['noise'] == manual_sum).all()

    down_result = exe.down(tree_result)
    assert (next(down_result.iter_leaves()).data['noise'] == manual_sum/2).all()


def test_lwte_levels_odd_tree(phony_executor):
    # build asymmetric tree
    root = JaxNode(children=[JaxNode()])

    left = JaxNode()
    left.children = [JaxNode(children=[JaxNode(), JaxNode()]) for _ in range(5)]    

    right = JaxNode()
    right.children = [JaxNode(children=[JaxNode()]) for _ in range(2)]    

    root.children += [left,right]

    tree = JaxTree(root)

    # test if execution order makes sense

    result = phony_executor._determine_execution_order(tree)

    sizes = [len(level) for level in result]

    assert sizes == [1,3,7,12]


def test_aggregator(phony_executor):
    ones = range(90)
    for x in phony_executor._batch_aggregate(ones):
        assert len(x) >= 10

#@pytest.mark.skip(reason='in development')
def test_heap_sum(noise_tree):
    assert len(noise_tree.root.children) == 2
    up = lambda noise, **kwargs: noise
    fuse = lambda messages, **kwags: {'noise' : messages.sum(0)}
    down = lambda parent_noise, **kwargs: {'noise': parent_noise/2}

    exe = DependencyTreeExecutor(up=up,down=down,fuse=fuse)

    manual_sum = jnp.stack([node.data['noise'] for node in noise_tree.iter_leaves()]).sum(0)

    tree_result = exe.up(noise_tree)

    assert (tree_result.root.data['noise'] == manual_sum).all()

    down_result = exe.down(tree_result)
    assert (next(down_result.iter_leaves()).data['noise'] == manual_sum/2).all()

