from hyperiax.execution import LevelwiseTreeExecutor, DependencyTreeExecutor
from hyperiax.models import UpDownLambda
from test_fixtures import small_tree, phony_executor,noise_tree
from hyperiax.models.functional import pass_up, sum_fuse_children
from hyperiax.tree import TreeNode, HypTree
import hyperiax
from jax.random import PRNGKey
from jax import numpy as jnp
import pytest

from hyperiax.tree.fasttree import FastTree, TopologyNode
from hyperiax.plotting.ascii import plot_tree_text
from hyperiax.models.lambdamodels import UpLambda
from hyperiax.execution import FastOrderedExecutor


def test_asym():
    root = TopologyNode()
    root.children = [TopologyNode(root) for _ in range(4)]
    root.children[1].children = [TopologyNode(root.children[1]) for _ in range(2)]
    root.children[3].children = [TopologyNode(root.children[3])]
    tree = FastTree(root)
    tree.add_property('count', (1,))
    tree.data['count'] = jnp.ones_like(tree.data['count'])
    def _up(count, **kwargs):
        return {'count': count}
    def _transform(child_count, count, **kwargs):
        return {'count': child_count+count}
    
    upmodel = UpLambda(_up, _transform, {'count': 'sum'})
    exe = FastOrderedExecutor(upmodel)
    res = exe.up(tree)

    assert (res['count'].squeeze() == jnp.array([8,1,3,1,2,1,1,1])).all()
    assert res['count'][0] == len(tree)

def test_bucket_truncate():
    a,b = FastTree._truncate_bucket(None, jnp.array([2,2,2,4,4,9,9,10,10]))

    assert (a == jnp.array([2,4,9,10])).all()
    assert (b == jnp.array([0,0,0,1,1,2,2,3,3])).all()


def test_lwte_sum(noise_tree):
    assert len(noise_tree.root.children) == 2
    up = pass_up('noise')
    fuse = sum_fuse_children(0)
    down = lambda parent_noise, **kwargs: {'noise': parent_noise/2}

    model = UpDownLambda(up, fuse, down)

    exe = LevelwiseTreeExecutor(model)

    manual_sum = jnp.stack([node['noise'] for node in noise_tree.iter_leaves()]).sum(0)
    tree_result = exe.up(noise_tree)

    assert (tree_result.root['noise'] == manual_sum).all()

    down_result = exe.down(tree_result)
    assert (next(down_result.iter_leaves())['noise'] == manual_sum/2).all()


def test_lwte_levels_odd_tree(phony_executor):
    # build asymmetric tree
    root = TreeNode(children=[TreeNode()])

    left = TreeNode()
    left.children = [TreeNode(children=[TreeNode(), TreeNode()]) for _ in range(5)]    

    right = TreeNode()
    right.children = [TreeNode(children=[TreeNode()]) for _ in range(2)]    

    root.children += [left,right]

    tree = HypTree(root)

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
    up = pass_up('noise')
    fuse = sum_fuse_children(0)
    down = lambda parent_noise, **kwargs: {'noise': parent_noise/2}


    model = UpDownLambda(up, fuse,  down)

    exe = LevelwiseTreeExecutor(model)

    manual_sum = jnp.stack([node.data['noise'] for node in noise_tree.iter_leaves()]).sum(0)

    tree_result = exe.up(noise_tree)

    assert (tree_result.root.data['noise'] == manual_sum).all()

    down_result = exe.down(tree_result)
    assert (next(down_result.iter_leaves()).data['noise'] == manual_sum/2).all()

