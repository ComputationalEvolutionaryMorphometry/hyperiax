import hyperiax
from jax.random import PRNGKey, split
from hyperiax.execution import LevelwiseTreeExecutor
from hyperiax.tree.fasttree import FastTree
import jax
from hyperiax.execution.collate import dict_collate
from hyperiax.tree.builders import symmetric_topology
import time
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
key = PRNGKey(0)

depths = list(range(1, 19, 1))
newt = []
oldt = []

for depth in tqdm(depths):

    topology = symmetric_topology(depth, 2)
    fast_tree = FastTree(topology)
    #print(f"Fast tree with {len(fast_tree)} nodes")

    fast_tree.add_property('noise', (10,2))


    start = time.time()
    for level_start, level_end in fast_tree.levels:
        # start by fusing children  
        data = jax.lax.slice_in_dim(fast_tree.data['noise'].data, level_start, level_end)
            

    end = time.time()
    #print(f"New execution time: {end-start}")
    newt.append(end-start)

    tree = hyperiax.tree.builders.symmetric_tree(depth,2)
    #print(f"Old tree with {len(tree)} nodes")
    subkey, key = split(key)
    noise_tree = hyperiax.tree.initializers.initialize_noise(tree, key, (10, 2))

    lwe = LevelwiseTreeExecutor(None, batch_size=5000)
    noise_tree.order = (lwe._determine_execution_order(noise_tree), None)
    order = reversed(noise_tree.order[0])

    start = time.time()
    for level in order:
        # start by fusing children  
        for nodes in lwe._batch_aggregate(level):
            data = dict_collate([node.data for node in nodes])
            

    end = time.time()
    #print(f"Old execution time: {end-start}")
    oldt.append(end-start)

plt.plot(depths, newt, label='New')
plt.plot(depths, oldt, label='Old')
plt.legend()
plt.xlabel('Depth')
plt.ylabel('Time (s)')
plt.yscale('log')
plt.savefig('batch_generation.png')