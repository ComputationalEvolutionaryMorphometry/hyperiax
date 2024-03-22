from typing import Iterable, List, SupportsIndex
from copy import deepcopy, copy

class ChildList(list):
    def __init__(self, children : List = []) -> None:
        super().__init__(children)

    def __setitem__(self, idx, node):
        raise ValueError('Cannot assign children directly, use `node.add_child(...)`')
    
    def append(self, node):
        raise ValueError('Cannot assign children directly, use `node.add_child(...)`')
    
    def __iadd__(self, __value: Iterable):
        raise ValueError('Cannot assign children directly, use `node.add_child(...)`')
    
    def _add_child(self, child):
        super().append(child)

    def __copy__(self):
        clone = copy(super())
        return ChildList(clone)
    
    def __deepcopy__(self, memo):
        clone = deepcopy(list(self), memo=memo)
        return ChildList(clone)
    
    def __bool__(self):
        return len(self) > 0