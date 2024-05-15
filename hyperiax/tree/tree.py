from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator, Callable


@dataclass(repr=False)
class TreeNode:
    """
    The data nodes within a tree

    """
    parent : TreeNode = None
    data : dict = field(default_factory=dict)
    children : list[TreeNode] = None
    name : str = None
   

    def __repr__(self) -> str:
        return f'TreeNode({self.data}) with {len(self.children)} children' if self.children else f'TreeNode({self.data}) with no children'

    def __getitem__(self, arg: Any) -> Any:
        return self.data.__getitem__(arg)

    def __setitem__(self, key: str, arg: Any) -> None:
        self.data.__setitem__(key, arg)

    def __delitem__(self, key: str) -> None:
        self.data.__delitem__(key)

    def add_child(self, child: TreeNode) -> TreeNode:
        """ 
        Add an individual child to the node

        :param child: child node to be added
        :return: the child node

        """
        child.parent = self
        self.children._add_child(child)
        return child



class HypTree:
    """
    The tree class that wraps behavious around a set of nodes.

    The set of nodes is given via the `root` node, and can be iterated conveniently using the utility in this class.

    """
    def __init__(self, root : TreeNode) -> None:
        self.root = root
        self.order = None
        
    def __repr__(self) -> str:
        return f'HypTree with {len(list(self.iter_levels()))} levels and {len(self)} nodes'

    def __len__(self) -> int:
        return len(list(self.iter_bfs()))
    
    def __getitem__(self, arg: Any) -> Iterator[Any]:
        for node in self.iter_bfs():
            yield node.data.__getitem__(arg)

    def __setitem__(self, key: str, arg: Any) -> None:
        if '__iter__' in getattr(arg, '__dict__', dict()):
            for node, value in zip(self.iter_bfs(), arg):
                node.data.__setitem__(key, value)
        else:
            for node in self.iter_bfs():
                node.data.__setitem__(key, arg)
    
    def copy(self) -> HypTree:
        """
        Returns a copy of the tree

        :return: a copy of the tree

        """
        return copy.deepcopy(self)

    def iter_leaves(self) -> Iterator[TreeNode]:
        """
        Iterate over all of the leaves in the tree

        """

        queue = deque([self.root])

        while queue:
            current = queue.popleft()
            if current.children:
                queue.extend(current.children)
            else:
                yield current

    def iter_bfs(self) -> Iterator[TreeNode]:
        """
        Iterate over all of the nodes in a breadth first manner

        """
        queue = deque([self.root])

        while queue:
            current = queue.popleft()
            if current.children:
                queue.extend(current.children)
            yield current

    def iter_dfs(self) -> Iterator[TreeNode]:
        """
        Iterate over all of the nodes in a depth-first manner.

        """
        stack = deque([self.root])

        while stack:
            current = stack.pop()
            if current.children:
                stack.extend(current.children)
            yield current

    def iter_levels(self) -> Iterator[list[TreeNode]]:
        """
        Iterate over each level in the tree

        """
        queue = deque()
        buffer_queue = deque([self.root])
        while queue or buffer_queue:
            if not queue: # if queue is empty, flush the buffer and yield a level
                queue = buffer_queue
                yield list(buffer_queue) # to not pass the reference
                buffer_queue = deque()

            if children := queue.popleft().children:
                buffer_queue.extend(children)

    def iter_leaves_dfs(self) -> Iterator[TreeNode]:
        """
        Iterates over the leaves in the tree using depth-first search.

        ??? Duplicate of iter_leaves
        """
        stack = deque([self.root])

        while stack:
            current = stack.pop()
            if current.children:
                stack.extend(reversed(current.children))
            else:
                yield current

    def plot_tree_2d(self, ax: Any=None, selector: Callable[[dict], Any]=None) -> None:
        """
        Visualize the tree data in 2D plane

        :param ax: the axis to plot the tree on, if None, a new figure is created, defaults to None
        :param selector: a function to select the specific data in the nodes to plot, if None, then all data is plotted, defaults to None
        """
        from .plot_utils import plot_tree_2d_
        plot_tree_2d_(self, ax, selector)

    def plot_tree(self, ax: Any=None, inc_names: bool=False) -> None:
        """
        Visualize the hierarchical structure of the tree

        :param ax: the axis to plot the tree on, if None, a new figure is created, defaults to None
        :param inc_names: whether to include the names of the nodes in the plot, defaults to False
        """
        tree = self.copy()
        from .plot_utils import plot_tree_
        plot_tree_(tree, ax,inc_names)

    def plot_tree_text(self) -> None:
        """
        Visualize the tree structure in the form of text, with minimal information except the structure
        """
        from .plot_utils import HypTreeFormatter
        formatter = HypTreeFormatter(self)
        formatter.print_tree()

    def to_newick(self) -> str:
        """
        Convert the tree structure into a Newick string

        :return: the Newick string representation of the tree structure
        """
        # Recursive function to convert tree to Newick string
        def node_to_newick(node:TreeNode) -> str:
            """
            Convert the tree Node to a Newick string

            :param node: the node to convert to Newick string
            :return: the Newick string representation of the node
            """
            if node is None:
                return ''
            
            parts = []
            if node.children is not None:
                for child in node.children:

                    part = node_to_newick(child)
                    parts.append(part)
                
            # If the current node has children, enclose the children's string representation in parentheses
            if parts:
                children_str = '(' + ','.join(parts) + ')'
            else:
                children_str = ''

            # Node name and distance formatting
            node_info = node.name if node.name else ''
            if 'edge_length' in node.data:
                node_info += ':' + str(node.data['edge_length'])

            # For nodes that have both children and their own information (name or distance)
            if children_str or node_info:
                return children_str + node_info
            else:
                # For the very rare case where a node might not have a name or children (unlikely in valid Newick)
                return ''

        return node_to_newick(self.root) + ';'
