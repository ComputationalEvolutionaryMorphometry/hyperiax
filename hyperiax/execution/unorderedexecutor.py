from ..models import UpdateReducer, UpdateModel
import jax
from inspect import getfullargspec
from functools import partial
from jax import numpy as jnp

class UnorderedExecutor:
    "Executes nodes in an unstructured way; giving full context from both parents and children"

    def __init__(self, model) -> None:
        self.model = model

    def update(self, tree, key, params = {}):
        if issubclass(type(self.model), UpdateReducer):
            new_data = self._update_reduce_inner(tree.data, tree, params, key)
        elif issubclass(type(self.model), UpdateModel):
            coloring = tree.coloring
            red_indx = jnp.arange(tree.size)[coloring]
            new_data = self._update_inner(tree.data, tree, red_indx, params)
            black_indx = jnp.arange(tree.size)[~coloring]
            new_data = self._update_inner(new_data, tree, black_indx, params)
        else:
            raise ValueError('Model needs to be of type UpdateReducer or UpdateModel')
        tree.data = {**tree.data, **new_data}

    @partial(jax.jit, static_argnames=['tree', 'self']) 
    def _update_inner(self, data, tree, coloring, params):
        node_data = {
            k: data[k][coloring]
            for k in self.model.update_current_keys
        }
        parent_data = {
            f'parent_{k}': data[k][tree.parents[coloring]]
            for k in self.model.update_parent_keys
        }
        child_data = {
            f'child_{k}': data[k][tree.gather_child_idx[coloring]]
            for k in self.model.update_child_keys
        }

        result = self.model.update(**parent_data, **node_data, **child_data, 
                        root_mask=tree.is_root[coloring], 
                        leaf_mask=tree.is_leaf[coloring], params = params)

        for k, val in result.items():
            data[k] = data[k].at[coloring].set(val)

        return data


    #@partial(jax.jit, static_argnames=['tree', 'self']) 
    def _update_reduce_inner(self, data, tree, params, key):
        iterator = zip(
            reversed(tree.levels[1:]), # indices for levels
            reversed(tree.pbuckets_ref) # indices for upwards propagation
        )
        unrolled = list(iterator)
        i1 = unrolled[::2]
        i2 = unrolled[1::2]
        # we can make this much faster by actually merging the underlying structures... #TODO
        key, k1 = jax.random.split(key)
        for it in (i1, i2):
            for (level_start, level_end), up_ref in it:
                child_node_data = {
                    k: jax.lax.slice_in_dim(data[k], level_start, level_end)
                    for k in self.model.update_child_keys
                }
                #print(child_node_data)
                up_result = self.model.up(**child_node_data, params = params)

                segments = tree.pbuckets[level_start:level_end]

                fuse_scatter = {
                    f'child_{k}': self.model.reductions[k](v, segments, num_segments=len(up_ref), indices_are_sorted=True)
                    for k,v in up_result.items()
                }

                node_data = {
                    k: data[k][up_ref]
                    for k in self.model.update_node_keys
                }

                parent_data = {
                    f'parent_{k}': data[k][tree.parents[up_ref]]
                    for k in self.model.update_parent_keys
                }
                print(tree.is_root[up_ref])
                print(tree.is_leaf[up_ref])

                result = self.model.update(
                    **parent_data, **node_data, **fuse_scatter, 
                    root_mask=tree.is_root[up_ref], 
                    leaf_mask=tree.is_leaf[up_ref],params = params, key=k1)
                for k, val in result.items():
                    data[k] = data[k].at[up_ref].set(val)


        return data