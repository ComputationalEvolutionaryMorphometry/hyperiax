from __future__ import annotations

from typing import Any, Iterable
from copy import deepcopy, copy

class ChildList(list):
    """ 
    A list for storing all children of a parent node, but prevents direct assignment.
    """
    def __init__(self, children : list = []) -> None:
        super().__init__(children)

    def __setitem__(self, idx, node):
        raise ValueError('Cannot assign children directly, use `node.add_child(...)`')
    
    def append(self, node):
        raise ValueError('Cannot assign children directly, use `node.add_child(...)`')
    
    def __iadd__(self, __value: Iterable):
        raise ValueError('Cannot assign children directly, use `node.add_child(...)`')
    
    def _add_child(self, child):
        super().append(child)

    def __copy__(self) -> ChildList:
        """ 
        Return a copy of the ChildList

        :return: A copy of the ChildList
        """
        clone = copy(super())
        return ChildList(clone)
    
    def __deepcopy__(self, memo: dict[int, Any] | None = None):
        """ 
        Return a deep copy of the ChildList

        :param memo: The memo dictionary, defaults to None
        :return: A deep copy of the ChildList
        """
        clone = deepcopy(list(self), memo=memo)
        return ChildList(clone)
    
    def __bool__(self) -> bool:
        """
        Check if the ChildList is not empty

        :return: True if the ChildList is not empty, False otherwise
        """
        return len(self) > 0