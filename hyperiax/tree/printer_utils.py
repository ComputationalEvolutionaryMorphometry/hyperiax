# Functions for nicer tree printing in the terminal, modified from https://github.com/AharonSambol/PrettyPrintTree

from typing import Any, List, Tuple, Generator

from hyperiax.tree.tree import TreeNode, HypTree

def zip_longest(*iterables: List, default: Any) -> Generator:
    """
    Returns an iterator that aggregates elements from each of the input iterables.
    If the iterables are of unequal length, missing values are filled with the specified default value.

    Args:
        *iterables: Variable number of iterables to be zipped together.
        default: The value to be used as a default when an iterable is exhausted.

    Yields:
        A tuple containing elements from each of the input iterables.

    Raises:
        None.

    Examples:
        >>> list(zip_longest([1, 2, 3], [4, 5], [6, 7, 8, 9], default=None))
        [(1, 4, 6), (2, 5, 7), (3, None, 8), (None, None, 9)]
    """
    lens = [len(x) for x in iterables]
    for line in range(max(lens)):
        yield tuple(
            iterable[line] if line < lens[i] else default
            for i, iterable in enumerate(iterables)
        )

def left_adjust(text: str, amount: int, padding: str = ' ') -> str:
    """
    Adjusts the given text to the left by adding padding characters.

    Args:
        text (str): The text to be adjusted.
        amount (int): The desired width of the adjusted text.
        padding (str, optional): The character used for padding. Defaults to ' '.

    Returns:
        str: The adjusted text.

    """
    return text + padding * (amount - len(text))

def right_adjust(text: str, amount: int, padding: str =' ') -> str:
    """
    Adjusts the given text to the right by adding padding characters.

    Args:
        text (str): The text to be right adjusted.
        amount (int): The total width of the resulting string.
        padding (str, optional): The character used for padding. Defaults to ' '.

    Returns:
        str: The right adjusted text.

    """
    return padding * (amount - len(text)) + text

class TreeNodeFormatter:
    """
    A class that represents a formatter for tree nodes.
    """

    @classmethod
    def from_string(cls: Any, string: str) -> Any:
        """
        Create a TreeNodeFormatter object from a string representation.

        Args:
            string (str): The string representation of the tree node.

        Returns:
            TreeNodeFormatter: The TreeNodeFormatter object.
        """
        lines = string.split('\n')
        height = len(lines)
        width = max(len(line) for line in lines)
        return cls(lines, height=height, width=width)
    
    def __init__(self, lines: List[str], *, height: int, width = int, middle_height: int = None, middle_width: int = None) -> None:
        """
        Initialize a TreeNodeFormatter object.

        Args:
            lines (list[str]): The content of the tree node.
            height (int): The height of the tree node text.
            width (int): The width of the tree node text.
            middle_height (int, optional): The middle height position of the tree node text. Defaults to None.
            middle_width (int, optional): The middle width position of the tree node text. Defaults to None.
        """
        self.lines = lines
        self.height = height
        self.width = width
        self._middle_height = middle_height
        self._middle_width = middle_width

    def to_string(self) -> str:
        """
        Convert the TreeNodeFormatter object to a string representation.

        Returns:
            str: The string representation of the tree node.
        """
        return '\n'.join(self.lines)
    
    @property
    def middle_width(self) -> int:
        """
        Get the middle width of the tree node text.

        Returns:
            int: The middle width position of the tree node text.
        """
        if self._middle_width is None:
            return sum(divmod(self.width, 2)) - 1
        return self._middle_width
    
    @property
    def middle_height(self) -> int:
        """
        Get the middle height of the tree node text.

        Returns:
            int: The middle height position of the tree node text.
        """
        if self._middle_height is None:
            return sum(divmod(self.height, 2)) - 1
        return self._middle_height

def add_parent(parent_formatter: TreeNodeFormatter, children_formatter: TreeNodeFormatter) -> TreeNodeFormatter:
    """
    Combines the formatting information of a parent node and its children nodes to create a new TreeNodeFormatter object.

    Args:
        parent_formatter (TreeNodeFormatter): The formatting information of the parent node.
        children_formatter (TreeNodeFormatter): The formatting information of the children nodes.

    Returns:
        TreeNodeFormatter: A new TreeNodeFormatter object with the combined formatting information.

    """
    parent_middle, children_middle = parent_formatter.middle_width, children_formatter.middle_width
    parent_width, children_width = parent_formatter.width, children_formatter.width
    
    if parent_middle == children_middle:      # Only one child
        lines = parent_formatter.lines + children_formatter.lines
        middle = parent_middle
    elif parent_middle < children_middle:     # Have several children
        padding = ' ' * (children_middle - parent_middle)
        lines = [padding + line for line in parent_formatter.lines] + children_formatter.lines
        parent_width += children_middle - parent_middle
        middle = children_middle
    else:                                    # Have no child
        padding = ' ' * (parent_middle - children_middle)
        lines = parent_formatter.lines + [padding + line for line in children_formatter.lines]
        children_width += parent_middle - children_middle
        middle = parent_middle
    
    return TreeNodeFormatter(
        lines,
        height=parent_formatter.height + children_formatter.height,
        width=max(parent_width, children_width),
        middle_width=middle
    )

class HypTreeFormatter:
    """
    A class that formats a HypTree object into a string representation.

    Attributes:
        root (TreeNode): The root node of the HypTree object.

    Methods:
        get_children(node: TreeNode) -> list[TreeNode]:
            Returns the children of a given node.
        
        get_name(node: TreeNode) -> str:
            Returns the name of a given node. If the name is None, returns '*'.
        
        add_name(name: str, node_formatter: TreeNodeFormatter, parent_adder: callable=add_parent, seperator: str = ' ') -> TreeNodeFormatter:
            Adds a name to the node formatter and returns the updated formatter.
        
        format() -> str:
            Formats the HypTree object into a string representation and returns it.
        
        tree_join_formatter(node: TreeNode, depth: int = 0) -> TreeNodeFormatter:
            Recursively formats the tree structure and returns the root node formatter.
        
        node_to_formatter(node: TreeNode) -> TreeNodeFormatter:
            Converts a node into a TreeNodeFormatter object and returns it.
        
        print_tree() -> None:
            Prints the formatted tree representation to the console.
    """

    def __init__(self, tree: HypTree) -> None:
        self.root = tree.root

    def get_children(self, node: TreeNode) -> List[TreeNode]:
        return node.children
    
    def get_name(self, node: TreeNode) -> str:
        return node.name if node.name else '*'
    
    # def add_name(self, name: str, node_formatter: TreeNodeFormatter, parent_adder: callable=add_parent, seperator: str = ' ') -> TreeNodeFormatter:
    #     if name:
    #         name_formatter = TreeNodeFormatter.from_string(str(name))
    #         node_formatter = parent_adder(TreeNodeFormatter.from_string(seperator), node_formatter)
    #         node_formatter = parent_adder(name_formatter, node_formatter)
    #     return node_formatter
    
    def format(self) -> str:
        """
        Formats the tree structure into a string representation.

        Returns:
            str: The formatted tree structure as a string.
        """
        res = self.tree_join_formatter(self.root)
        res = res.to_string().rstrip()
        return res
    
    def tree_join_formatter(self, node: TreeNode, depth: int = 0) -> TreeNodeFormatter:
        """
        Recursively formats the tree structure and returns the root node formatter.

        Args:
            node (TreeNode): The current node being formatted.
            depth (int): The depth of the current node in the tree structure.

        Returns:
            TreeNodeFormatter: The root node formatter.
        """
        # name = self.get_name(node)
        children = self.get_children(node)
        node_formatter = self.node_to_formatter(node)

        if children:
            children_formatters = [
                self.tree_join_formatter(child, depth=depth+1)
                for child in children
            ]
            if len(children) == 1:
                children_node_formatter = children_formatters[0]
                children_node_formatter.lines.insert(0, ' ' * children_node_formatter.middle_width + '│')
            else:
                children_node_formatter = join_horizontally(children_formatters)
            
            node_formatter = add_parent(node_formatter, children_node_formatter)
        
        # node_formatter = self.add_name(name, node_formatter)
            
        return node_formatter
    
    def node_to_formatter(self, node: TreeNode) -> TreeNodeFormatter:
        """
        Converts a node into a TreeNodeFormatter object and returns it.

        Args:
            node (TreeNode): The node to convert.

        Returns:
            TreeNodeFormatter: The converted node formatter.
        """
        name = self.get_name(node)
        return TreeNodeFormatter.from_string(str(name))
    
    def print_tree(self) -> None:
        """
        Prints the formatted tree representation to the console.
        """
        print(self.format())

def join_horizontally(boxes: List[TreeNodeFormatter]) -> TreeNodeFormatter:
    """
    Joins multiple TreeNodeFormatter boxes horizontally to form a line to print.

    Args:
        boxes (list[TreeNodeFormatter]): A list of TreeNodeFormatter objects to gather.

    Returns:
        TreeNodeFormatter: The joined TreeNodeFormatter box.

    """
    lines, width, height = join_boxes(boxes)
    middle = add_pipes(boxes, lines)
    height += 1
    return TreeNodeFormatter(lines, height=height, width=width, middle_width=middle)

def join_boxes(boxes: List[TreeNodeFormatter]) -> Tuple[List[str], int, int]:
    """
    Joins multiple TreeNodeFormatter together horizontally to get the text, width and height of this line.

    Args:
        boxes (List[TreeNodeFormatter]): A list of TreeNodeFormatter objects to gather.

    Returns:
        Tuple[List[str], int, int]: A tuple containing the joined lines, width, and height of the joined boxes.
    """
    lines = [
        ' '.join(
            left_adjust(text=line, amount=boxes[i].width)
            for i, line in enumerate(lines)
        )
        for lines in zip_longest(*(box.lines for box in boxes), default='')
    ]
    width = sum(box.width for box in boxes) + len(boxes) - 1
    height = max(box.height for box in boxes)
    return lines, width, height

def add_pipes(boxes: List[TreeNodeFormatter], lines: List[str]) -> int:
    """
    Adds pipes to the given lines to create a tree-like structure.

    Args:
        boxes (List[TreeNodeFormatter]): A list of TreeNodeFormatter objects to gather.
        lines (List[str]): A list of strings representing the lines of the tree.

    Returns:
        int: The number of characters added as padding to the lines.

    """
    padding = ' ' * boxes[0].middle_width
    pipes = '┌'
    for prev, box in zip(boxes, boxes[1:]):
        pipes += '─' * (prev.width - prev.middle_width + box.middle_width) + '┬'
    middle_of_pipes = sum(divmod(len(pipes), 2)) - 1
    pipes = (
        padding 
        + pipes[:middle_of_pipes] 
        + {"─": "┴", "┬": "┼", "┌": "├", "┐": "┤"}[pipes[middle_of_pipes]]
        + pipes[middle_of_pipes + 1:-1]
        + '┐'
    )
    lines.insert(0, pipes)
    return len(padding) + middle_of_pipes