from __future__ import annotations
from abc import ABC
from ..tree.fasttree import FastTree
import jax
from ..models import UpModel, UpDownModel, DownModel, FuseModel
from inspect import getfullargspec
from functools import partial

class OrderedExecutor(ABC):
    """Abstract for the ordered executor.

    Ordered executors performs operations in a certain direction (eg. up/down).

    It contains on abstract method `_determine_execution_order``, which gives the order in which nodes
    are to be executed in the tree. It is assumed that all operations are batched.
    """
    def __init__(self, 
                 model : UpModel | UpDownModel | DownModel,
                 key = None) -> None:
        self.key = key if key else jax.random.PRNGKey(0)
        self.model = model

        if issubclass(type(self.model), (UpDownModel, DownModel)):
            self._set_down_keys()
        if issubclass(type(self.model), (UpDownModel, UpModel)):
            self._set_up_keys()
        if issubclass(type(self.model), FuseModel):
            self._set_fuse_keys()

    def updown(self, tree, params = {}):
        """Runs both the up and downward pass

        Args:
            tree (HypTree): The tree to execute on
            params (dict, optional): optional parameters for mcmc. Defaults to {}.

        Returns:
            HypTree: The tree after running one up and down pass
        """
        t = self.up(tree, params=params)
        t = self.down(t, params=params)

        return t
    

    def up(self, tree : FastTree, params = {}):
        if not issubclass(type(self.model), (UpDownModel, UpModel)):
            raise ValueError('Model needs to be of type UpDownModel or UpModel')
        # out here we fetch the data stored in the tree

        #for k,val_shape in self.model.up_produces.items():
        #    if k not in tree.data:
        #        tree.add_property(k, val_shape)

        if issubclass(type(self.model), (UpDownModel, UpModel)):
            new_data = self._up_inner(tree.data, tree, params)

            tree.data = {**tree.data, **new_data}

            return new_data
        elif issubclass(type(self.model), FuseModel):
            new_data = self._fuse(tree, params = params)

            tree.data = {**tree.data, **new_data}

            return new_data

    @partial(jax.jit, static_argnames=['tree', 'self']) 
    def _up_inner(self, data, tree, params):
        iterator = zip(
            reversed(tree.levels[1:]), # indices for levels
            #reversed(tree.psizes[:-1]), # indices for grouping calculations
            reversed(tree.pbuckets_ref) # indices for upwards propagation
        )
        for (level_start, level_end), up_ref in iterator:
            node_data = {
                k: jax.lax.slice_in_dim(data[k], level_start, level_end)
                for k in self.up_keys
            }
            #print(node_data)
            up_result = self.model.up(**node_data, params = params)

            segments = tree.pbuckets[level_start:level_end]

            fuse_scatter = {
                f'child_{k}': self.model.reductions[k](v, segments, num_segments=len(up_ref), indices_are_sorted=True)
                for k,v in up_result.items()
            }

            parent_data = {
                k: data[k][up_ref]
                for k in self.transform_parent_keys
            }

            result = self.model.transform(**parent_data, **fuse_scatter, params = params)
            for k, val in result.items():
                data[k] = data[k].at[up_ref].set(val)


        return data

            


    def _set_up_keys(self):
        up_arg_spec = getfullargspec(self.model.up)
        upkeys = up_arg_spec.args
        self.up_keys = upkeys

        transform_arg_spec = getfullargspec(self.model.transform)
        keys = transform_arg_spec.args
        transform_child_keys = [k.removeprefix('child_') for k in keys if k.startswith('child_')]
        transform_parent_keys = [k for k in keys if not k.startswith('child_')]

        self.transform_child_keys = transform_child_keys
        self.transform_parent_keys = transform_parent_keys


    def _set_fuse_keys(self): 
        fuse_arg_spec = getfullargspec(self.model.transform)
        keys = fuse_arg_spec.args
        fuse_child_keys = [k.removeprefix('child_') for k in keys if k.startswith('child_')]
        fuse_parent_keys = [k for k in keys if not k.startswith('child_')]

        self.fuse_child_keys = fuse_child_keys
        self.fuse_parent_keys = fuse_parent_keys

    def _set_down_keys(self):
        arg_spec = getfullargspec(self.model.down)

        keys = arg_spec.args # TODO: we should remove things like key, up_msg, params

        down_parent_keys = [k.removeprefix('parent_') for k in keys if k.startswith('parent_')]
        down_child_keys = [k for k in keys if not k.startswith('parent_')]

        self.down_parent_keys = down_parent_keys
        self.down_child_keys = down_child_keys

    def _fuse(self, tree : FastTree, params = {}):
        if not issubclass(type(self.model), FuseModel):
            raise ValueError('Model needs to be of type FuseModel')
        # out here we fetch the data stored in the tree

        new_data = self._fuse_inner(tree.data, tree, params)

        tree.data = {**tree.data, **new_data}

        return tree

    @partial(jax.jit, static_argnames=['tree', 'self']) 
    def _down_inner(self, data, tree): #inner jitted runner for MAXIMUM PERFORMANCE
        #print(data, tree.levels)
        for level_start, level_end in tree.levels[1:]:
            node_data = {
                k: jax.lax.slice_in_dim(data[k], level_start, level_end)
                for k in self.down_child_keys
            }
            parent_indices = tree.parents[level_start:level_end] # no need to lax since tree is untraced
            parent_data = {
                f'parent_{k}': data[k][parent_indices]
                for k in self.down_parent_keys
            }

            result = self.model.down(**node_data, **parent_data)

            for k, val in result.items():
                data[k] = data[k].at[level_start:level_end].set(val)
        return data
    
    def down(self, tree : FastTree, params = {}):
        """Runs the down pass on the tree using the `down` function from the model.

        The `down` function in the model takes the following special parameters;

        `up_msg`: The message that was sent UP the edge during an upward pass
        `key`: A PRNGKey
        `params`: The same params passed to this function
        `parent_*`: Data from the parent node will be passed as arguments prefixed by `parent`.\
              If you have `value` stored in the parent node, `parent_value` will be an argument passed.
        `*`: Data from the child node will be passed as arguments without prefix. \
            If you have `value` stored in the child node, `value` will be an argument passed.

        The executor expects the `down` function to return a dictionary, \
            where each key returned overwrites the data in the lower node (with values not prefixed by `parent_`).
        Because of this, all values returned from `down` functions must have a leading axis of size `b`.

        Args:
            tree (HypTree): The tree to execute on
            params (dict, optional): optional parameters for mcmc. Defaults to {}.

        Returns:
            HypTree: The tree after running one pass
        """
        if not issubclass(type(self.model), (UpDownModel, DownModel)):
            raise ValueError('Model needs to be of type UpDownModel or DownModel')
        # out here we fetch the data stored in the tree

        new_data = self._down_inner(tree.data, tree)

        tree.data = {**tree.data, **new_data}

        return 
    
    def __call__(self, tree, params = {}):
        return self.updown(tree, params = params)