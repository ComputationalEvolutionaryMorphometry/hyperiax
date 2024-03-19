from typing import Any

from ..tree import TreeNode
from ..models import UpdateModel
from ..tree import HypTree
from abc import abstractmethod
from .collate import dict_collate

class UnorderedExecutor:
    "Executes nodes in an unstructured way; giving full context from both parents and children"

    def __init__(self, model : UpdateModel) -> None:
        self.model = model
        self.iterator_states = {}

    @abstractmethod
    def _determine_execution_pools(self, tree : HypTree):
        ...

    @abstractmethod
    def _iter_pools(self, pools):
        ...

    def get_iterator(self, tree : HypTree):
        """Get the iterator that runs over the tree

        Args:
            tree (HypTree): the tree to run over

        Returns:
            iterable: the iterator
        """
        pools = self._determine_execution_pools(tree)
        pooliter = iter(self._iter_pools(pools))
        return UpdateIterator(pooliter)

    def update(self, node : TreeNode, params):
        """This method executes the model implementation on the given node using the sampled params

        Args:
            node (TreeNode): The node to execute on
            params (dict): the sampled params to use

        Returns:
            bool: whether or not the sampled parameters should be accepted.
        """
        child_data = dict_collate([child.data for child in node.children]) if node.children else {}
        parent = node.parent.data if node.parent else {}

        new_vals, accept = self.model.update(
            parent_value=parent, 
            children_values=child_data, 
            node_value = node.data,
            parameters = params
        )

        node.data = {**node.data, **new_vals}

        return accept


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

class UpdateIterator:
    """Does nothing, but name is used to disambiguate"""
    def __init__(self, pool_iter) -> None:
        self._iter_internal = pool_iter
    def __iter__(self):
        iter(self._iter_internal)
        return self
    
    def __next__(self):
        return next(self._iter_internal)