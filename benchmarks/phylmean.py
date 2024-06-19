import hyperiax
from jax.random import PRNGKey, split
import jax
from jax import numpy as jnp
from hyperiax.execution import LevelwiseTreeExecutor, DependencyTreeExecutor
from hyperiax.models import UpLambda, DownLambda
from hyperiax.models.functional import pass_up
import jax
from functools import partial
from hyperiax.models.prebuilts import PhyloMeanModel

import time

key = PRNGKey(0)

levels = 13

tree = hyperiax.tree.builders.symmetric_tree(levels,2)
print(f"Got tree with {len(tree)} nodes")
subkey, key = split(key)
noise_tree = hyperiax.tree.initializers.initialize_noise(tree, key, (2,))
### dummy pass_up function (for testing)

for i, node in enumerate(noise_tree.iter_bfs()):
    key, subkey = split(key)
    node['edge_length'] = jax.random.uniform(subkey, (1,))

@jax.jit
def down(noise, edge_length,parent_value, **args):
    return {'value': jnp.sqrt(edge_length)*noise+parent_value}

noise_tree.root['value'] = noise_tree.root['noise']

downmodel = DownLambda(down_fn=down)
exe = DependencyTreeExecutor(downmodel, batch_size=5000)
stoc_tree = exe.down(noise_tree)

for node in stoc_tree.iter_leaves():
    node['estimated_value'] = node['value']

pme = PhyloMeanModel()
exe = DependencyTreeExecutor(pme, batch_size=5000)

print("going")
prebuilt_tree = exe.up(stoc_tree)
print("done")