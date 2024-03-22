from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List
from .childrenlist import ChildList


@dataclass(repr=False)
class TreeNode:
    """
    The data nodes within a tree, stupid by design. 
    Functionality is in JaxTree    
    """
    parent : TreeNode = None
    data : Dict = field(default_factory=dict)
    children : List[TreeNode] = None
    name : str = None
   

    def __repr__(self):
        return f'TreeNode({self.data}) with {len(self.children)} children' if self.children else f'TreeNode({self.data}) with no children'

    def __getitem__(self, arg):
        return self.data.__getitem__(arg)

    def __setitem__(self, key, arg):
        self.data.__setitem__(key, arg)

    def __delitem__(self, key):
        self.data.__delitem__(key)

    def add_child(self, child):
        child.parent = self
        self.children._add_child(child)
        return child



class HypTree:
    """The tree class that wraps behavious around a set of nodes.

    The set of nodes is given via the `root` node, and can be iterated conveniently using the utility in this class.
    """
    def __init__(self, root : TreeNode) -> None:
        self.root = root
        self.order = None
        
    def __repr__(self) -> str:
        return f'HypTree with {len(list(self.iter_levels()))} levels and {len(self)} nodes'

    def __len__(self) -> int:
        return len(list(self.iter_bfs()))
    
    def __getitem__(self, arg):
        for node in self.iter_bfs():
            yield node.data.__getitem__(arg)

    def __setitem__(self, key, arg):
        if '__iter__' in getattr(arg, '__dict__', dict()):
            for node, value in zip(self.iter_bfs(), arg):
                node.data.__setitem__(key, value)
        else:
            for node in self.iter_bfs():
                node.data.__setitem__(key, arg)
    
    def copy(self):
        return copy.deepcopy(self)

    def iter_leaves(self):
        """Iterates over the leaves in the tree

        Yields:
            iterator: an interator that runs over the leaves.
        """
        queue = deque([self.root])

        while queue:
            current = queue.popleft()
            if current.children:
                queue.extend(current.children)
            else:
                yield current

    def iter_bfs(self):
        """Iterate over all of the nodes in a breadth first manner

        Yields:
            iterator: an iterator that runs over the nodes
        """
        queue = deque([self.root])

        while queue:
            current = queue.popleft()
            if current.children:
                queue.extend(current.children)
            yield current

    def iter_levels(self):
        """Iterate over each level in the tree

        Yields:
            iterator: an iterator that runs over each level
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

    def plot_tree_2d(self, ax=None, selector=None):
        from matplotlib import pyplot as plt
        from matplotlib import patches as mpatch

        tree = 'partial'
        for node in self.iter_bfs():
            if node.data == None: break
        else:
            tree = 'full'

        cmap = plt.cm.ocean

        if ax == None:
            fig,ax = plt.subplots(figsize=(10,8))
        if tree == 'full':
            levels = list(self.iter_levels())

            for i, level in enumerate(levels):
                for node in level:
                    dat = selector(node.data) if selector else node.data
                    if node.children:
                        for child in node.children:
                            cdat = selector(child.data) if selector else child.data
                            ax.arrow(*dat, *(cdat-dat), width=0.01, length_includes_head=True, color='gray')
                    ax.scatter(*dat, color=cmap(i/len(levels)))
                    if 'name' in node.data.keys():
                        ax.annotate(node.data['name'], dat, xytext=(5,5), textcoords='offset pixels')

            handles = [mpatch.Patch(color=cmap(i/len(levels)), label = f'{i+1}') for i in range(len(levels))]
            legend = ax.legend(handles=handles, title="Levels")
            ax.add_artist(legend)
            ax.grid(True)

    def plot_tree(self, ax=None):
        import matplotlib.pyplot as plt
        def get_depth(node):
            """Recursively find the maximum depth of the tree."""
            if not hasattr(node, 'children') or not node.children:
                return 0
            return 1 + max(get_depth(child) for child in node.children)

        def plot_node(node, depth, x=0, y=0, dx=1, parent_pos=None):
            """Plot a single node and its children, spreading them equally."""
            plt.plot(x, y, 'ko')  # Plot the current node
            if parent_pos:
                plt.plot([parent_pos[0], x], [parent_pos[1], y], 'k-')  # Draw line to parent

            if hasattr(node, 'children') and node.children:
                n_children = len(node.children)
                width = dx * (2 ** (max_depth - depth - 1))  # Calculate spread based on depth
                start_x = x - width / 2 + width / (2 * n_children)  # Starting x position for children
                for i, child in enumerate(node.children):
                    child_x = start_x + i * (width / n_children)
                    plot_node(child, depth + 1, child_x, y - vertical_spacing, dx, (x, y))

        max_depth = get_depth(self.root)
        vertical_spacing = 1.5  # Vertical spacing between levels

        plt.figure(figsize=(10, 8))
        plot_node(self.root, 0, 0, max_depth * vertical_spacing, dx=1)
        plt.axis('off')
        plt.show()

    def plot_tree_text(self):
        from .printer_utils import HypTreeFormatter
        formatter = HypTreeFormatter(self)
        formatter.print_tree()

    def to_newick(self):
        # Recursive function to convert tree to Newick string
        def to_newick(node:TreeNode):
            """Recursively generates a Newick string from a JaxNode tree."""
            if node is None:
                return ''
            
            parts = []
            if node.children is not None:
                for child in node.children:

                    part = to_newick(child)
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

        """Converts a JaxTree to a Newick string."""
        return  to_newick(self.root) + ';'
