
from hyperiax.models.functional import pass_up, sum_fuse_children
from hyperiax.tree import TopologyNode, HypTree
from jax import numpy as jnp

from hyperiax.models.lambdamodels import UpLambdaReducer
from hyperiax.execution import OrderedExecutor
from hyperiax.plotting import plot_tree_text


def test_asym():
    root = TopologyNode()
    root.children = [TopologyNode(root) for _ in range(4)]
    root.children[1].children = [TopologyNode(root.children[1]) for _ in range(2)]
    root.children[3].children = [TopologyNode(root.children[3])]
    tree = HypTree(root)
    tree.add_property('count', (1,))
    tree.add_property('sum_count', (1,))
    tree.data['count'] = jnp.ones_like(tree.data['count'])
    tree.data['sum_count'] = tree.data['sum_count'].at[tree.is_leaf].set(tree.data['count'][tree.is_leaf])
    def _up(sum_count, **kwargs):
        return {'sub_count': sum_count}
    def _transform(child_sub_count, count, **kwargs):
        return {'sum_count': child_sub_count+count}
    
    print()
    plot_tree_text(tree)
    
    upmodel = UpLambdaReducer(_up, _transform, {'sub_count': 'sum'})
    exe = OrderedExecutor(upmodel)
    exe.up(tree)

    assert (tree.data['sum_count'].squeeze() == jnp.array([8,1,3,1,2,1,1,1])).all()
    assert tree.data['sum_count'][0] == len(tree)

def test_lwte_sum(noise_tree):
    assert len(noise_tree.topology_root.children) == 2
    up = lambda noise, **kwargs: {'noise':noise}
    def transform(child_noise, noise,**kwargs): 
        return {'noise': noise+child_noise}

    model = UpLambdaReducer(up, transform_fn=transform, reductions={'noise': 'sum'})

    exe = OrderedExecutor(model)

    manual_sum = noise_tree.data['noise'].sum(axis=0)
    exe.up(noise_tree)

    assert jnp.isclose(noise_tree.data['noise'][0], manual_sum).all()

    down_result = exe.down(tree_result)
    assert (next(down_result.iter_leaves())['noise'] == manual_sum/2).all()

    #print


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

