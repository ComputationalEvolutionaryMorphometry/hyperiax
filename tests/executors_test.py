
from hyperiax.models.functional import pass_up, sum_fuse_children
from hyperiax.tree import TopologyNode, HypTree
from jax import numpy as jnp

from hyperiax.models.lambdamodels import UpLambdaReducer
from hyperiax.execution import OrderedExecutor
from hyperiax.plotting import plot_tree_text
from hyperiax.models.lambdamodels import UpLambda, UpLambdaReducer
from test_fixtures import noise_tree


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

    #print

