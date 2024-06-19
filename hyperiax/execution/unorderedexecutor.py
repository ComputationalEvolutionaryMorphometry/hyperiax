from typing import Any

from ..tree import TreeNode
from ..models import UpdateModel
from ..tree import HypTree
from abc import abstractmethod
from .collate import dict_collate
import jax
from inspect import getfullargspec
from functools import partial

class UnorderedExecutor:
    "Executes nodes in an unstructured way; giving full context from both parents and children"

    def __init__(self, model : UpdateModel, key = None) -> None:
        self.model = model

        self.key = key if key else jax.random.PRNGKey(0)
        self.model = model

        self._set_update_keys()

    def _set_update_keys(self):
        up_arg_spec = getfullargspec(self.model.up)
        upkeys = up_arg_spec.args
        self.up_keys = upkeys

        update_arg_spec = getfullargspec(self.model.update)
        keys = update_arg_spec.args
        update_parent_keys = [k.removeprefix('parent_') for k in keys if k.startswith('parent_')]
        update_node_keys = [k for k in keys if not k.startswith('parent_') and not k.startswith('child_')]

        self.update_parent_keys = update_parent_keys
        self.update_node_keys = update_node_keys

    def update(self, tree, params = {}):
        if not issubclass(type(self.model), (UpdateModel,)):
            raise ValueError('Model needs to be of type UpdateModel')
        new_data = self._update_inner(tree.data, tree, params)

        tree.data = {**tree.data, **new_data}

        return new_data
    
    @partial(jax.jit, static_argnames=['tree', 'self']) 
    def _update_inner(self, data, tree, params):
        iterator = zip(
            reversed(tree.levels[1:]), # indices for levels
            reversed(tree.pbuckets_ref) # indices for upwards propagation
        )
        unrolled = list(iterator)
        i1 = unrolled[::2]
        i2 = unrolled[1::2]
        # we can make this much faster by actually merging the underlying structures... #TODO
        for it in (i1, i2):
            for (level_start, level_end), up_ref in it:
                child_node_data = {
                    k: jax.lax.slice_in_dim(data[k], level_start, level_end)
                    for k in self.up_keys
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
                    for k in self.update_node_keys
                }

                parent_data = {
                    f'parent_{k}': data[k][tree.parents[up_ref]]
                    for k in self.update_parent_keys
                }

                result = self.model.update(**parent_data, **node_data, **fuse_scatter, params = params)
                for k, val in result.items():
                    data[k] = data[k].at[up_ref].set(val)


        return data