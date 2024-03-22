from __future__ import annotations
from abc import ABC, abstractmethod
import copy
from itertools import islice
from ..tree import HypTree
import jax
from .collate import dict_collate, DictTransposer
from ..models import UpModel, UpDownModel, DownModel

class OrderedExecutor(ABC):
    """Abstract for the ordered executor.

    Ordered executors performs operations in a certain direction (eg. up/down).

    It contains on abstract method `_determine_execution_order``, which gives the order in which nodes
    are to be executed in the tree. It is assumed that all operations are batched.
    """
    def __init__(self, 
                 model : UpModel | UpDownModel | DownModel,
                 batch_size = 12,
                 key = None) -> None:
        self.batch_size = batch_size
        self.key = key if key else jax.random.PRNGKey(0)
        self.model = model

    @abstractmethod
    def _determine_execution_order(self, tree):
        """
            Determine the execution order for a downward pass.
            Upward pass will have the reverse order
        """

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
    
    def _batch_aggregate(self, iterator):
        it = iter(iterator)
        while True:
            batch = list(islice(it, self.batch_size))
            if not batch:
                return
            yield batch

    def up(self, tree : HypTree, params = {}):
        """Runs the up pass on the tree using the `up` and `fuse` functions from the model.

        The `up` function in the model takes the following special parameters;
        `key`: A PRNGKey
        `params`: The same params passed to this function
        `*`: Data from the node will be passed as arguments without prefix batched according to `batch_size`. \
            If you have `value` stored in the node, `value` will be an argument passed batched with a leading axis of size `b`.

        The executor expects the `up` function to return a dictionary, \
            where each key returned overwrites the data in the `up` node.
        Because of this, all values returned from `up` functions must have a leading axis of size `b`.

        
        The `fuse` function in the model is not batched and takes the following special parameters;  
        `*`: Data from the node will be passed as arguments without prefix. \
            If you have `value` stored in the node, `value` will be an argument passed.
        `child_*`: Data from the `c` children nodes will be passed as arguments prefixed by `child`.\
            If you have `value` stored in the children nodes `child_value` will be passed as an argument to `fuse`,\
            being a stacked array with first axis of size `c`.
        The executor expects the `fuse` function to return a dictionary, \
            where each key returned overwrites the data in the node we are fusing.
            
        Args:
            tree (HypTree): The tree to execute on
            params (dict, optional): optional parameters for mcmc. Defaults to {}.

        Raises:
            ValueError: Mismatch on the number of returned items from client code. This happens when the model returns an incorrect number of items

        Returns:
            HypTree: The tree after running one pass
        """
        if not issubclass(type(self.model), (UpDownModel, UpModel)):
            raise ValueError('Model needs to be of type UpDownModel or UpModel')
        if not tree.order or tree.order[1] != type(self):
            tree.order = (self._determine_execution_order(tree), type(self))

        new_tree = copy.deepcopy(tree)
        
        order = reversed(new_tree.order[0])

        for level in order:
            # start by fusing children  
            for nodes in self._batch_aggregate(level):
                for node in nodes:
                    if not node.children: continue
                    fuse_data = dict_collate([child.up_val for child in node.children])
                    fuse_data = {f'child_{k}':v for k,v in fuse_data.items()}
                    fuse_result = self.model.fuse(**node.data, **fuse_data)

                    node.data = {**node.data, **fuse_result}

                data = dict_collate([node.data for node in nodes])

                self.key, subkey = jax.random.split(self.key)

                results = self.model.up(**data, key=subkey, params=params)
                iterator = zip(nodes, DictTransposer(results), strict = True)
                try:
                    for node,node_result in iterator:
                        node.up_val = node_result
                except ValueError:
                    raise ValueError(f"Number of returned elements ({max(len(v) for k,v in results.items())}) does not match number of nodes ({len(nodes)})")

        return new_tree


    
    def down(self, tree, params = {}):
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
        if not tree.order or tree.order[1] != type(self):
            tree.order = (self._determine_execution_order(tree), type(self))

        new_tree = copy.deepcopy(tree)
        
        order = new_tree.order[0]
        
        for level in order:
            for nodes in self._batch_aggregate(level):
                data = dict_collate([node.data for node in nodes])
                parents = [node.parent.data for node in nodes if node.parent]
                if len(parents) == len(nodes):
                    parents = dict_collate(parents)
                else:
                    parents = None
                if parents == None: continue

                upvals = [node.up_val for node in nodes if 'up_val' in node.__dict__ and node.up_val != None]
                if len(upvals) == len(nodes):
                    upvals = dict_collate(upvals)
                else:
                    upvals = None

                self.key, subkey = jax.random.split(self.key)
                pdict = {f'parent_{k}':v for k,v in parents.items()}
                results = self.model.down(**data, **pdict, up_msg=upvals, key=subkey, params=params) # 

                # results is expected to be a pytree with [b, ...] shape

                iterator = zip(nodes, DictTransposer(results), strict=True)
                try:
                    for node,node_result in iterator:
                        node.down_val = node_result

                        node.data = {**node.data, **node_result}
                except ValueError:
                    raise ValueError(f"Number of returned elements ({max(len(v) for k,v in results.items())}) does not match number of nodes ({len(nodes)})")


        return new_tree

    def __call__(self, tree, params = {}):
        return self.updown(tree, params = params)