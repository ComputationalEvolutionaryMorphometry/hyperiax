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
        """Runs the up pass on the tree

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
        """Runs the down pass on the tree

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

                iterator = zip(nodes, DictTransposer(results))
                for node,node_result in iterator:
                    node.down_val = node_result

                    node.data = {**node.data, **node_result}



        return new_tree

    def __call__(self, tree, params = {}):
        return self.updown(tree, params = params)