from typing import Any, Generator
from cmd2.ansi import style_aware_wcswidth as text_width

from hyperiax.tree.tree import TreeNode, HypTree

def zip_longest(*iterables: list, default: Any):
    lens = [len(x) for x in iterables]
    for line in range(max(lens)):
        yield tuple(
            iterable[line] if line < lens[i] else default
            for i, iterable in enumerate(iterables)
        )

def left_adjust(text: str, amount: int, padding: str = ' ') -> str:
    return text + padding * (amount - text_width(text))

def right_adjust(text: str, amount: int, padding: str =' ') -> str:
    return padding * (amount - text_width(text)) + text

class TreeNodeFormatter:
    
    @classmethod
    def from_string(cls: Any, string: str) -> Any:
        lines = string.split('\n')
        height = len(lines)
        width = max(text_width(line) for line in lines)
        return cls(lines, height=height, width=width)
    
    def __init__(self, lines: list[str], *, height: int, width = int, middle_height: int = None, middle_width: int = None) -> None:
        self.lines = lines
        self.height = height
        self.width = width
        self._middle_height = middle_height
        self._middle_width = middle_width

    def to_string(self) -> str:
        return '\n'.join(self.lines)
    
    @property
    def middle_width(self) -> int:
        if self._middle_width is None:
            return sum(divmod(self.width, 2)) - 1
        return self._middle_width
    
    @property
    def middle_height(self) -> int:
        if self._middle_height is None:
            return sum(divmod(self.height, 2)) - 1
        return self._middle_height

def add_parent(parent_formatter: TreeNodeFormatter, children_formatter: TreeNodeFormatter) -> TreeNodeFormatter:
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
    def __init__(self, tree: HypTree) -> None:
        self.root = tree.root

    def get_children(self, node: TreeNode) -> list[TreeNode]:
        return node.children
    
    def get_name(self, node: TreeNode) -> str:
        return node.name if node.name else '*'
    
    def add_name(self, name: str, node_formatter: TreeNodeFormatter, parent_adder: callable=add_parent, seperator: str = ' ') -> TreeNodeFormatter:
        if name:
            name_formatter = TreeNodeFormatter.from_string(str(name))
            node_formatter = parent_adder(TreeNodeFormatter.from_string(seperator), node_formatter)
            node_formatter = parent_adder(name_formatter, node_formatter)
        return node_formatter
    
    def format(self) -> str:
        res = self.tree_join_formatter(self.root)
        res = res.to_string().rstrip()
        return res
    
    def tree_join_formatter(self, node: TreeNode, depth: int = 0) -> TreeNodeFormatter:
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
        name = self.get_name(node)
        return TreeNodeFormatter.from_string(str(name))
    
    def print_tree(self) -> None:
        print(self.format())

def join_horizontally(boxes: list[TreeNodeFormatter]) -> TreeNodeFormatter:
    lines, width, height = join_boxes(boxes)
    middle = add_pipes(boxes, lines)
    height += 1
    return TreeNodeFormatter(lines, height=height, width=width, middle_width=middle)

def join_boxes(boxes: list[TreeNodeFormatter]) -> tuple[list[str], int, int]:
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

def add_pipes(boxes: list[TreeNodeFormatter], lines: list[str]) -> int:
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