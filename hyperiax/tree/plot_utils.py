# Functions for plotting data and tree
import numpy as np

#####################################################################################################
# 2d plot of data points in a 

def plot_tree_2d_(self, ax=None, selector=None):
    """
    Visualize the tree data in 2D plane

    :param ax: the axis to plot the tree on, if None, a new figure is created, defaults to None
    :param selector: a function to select the specific data in the nodes to plot, if None, then all data is plotted, defaults to None
    """
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

#####################################################################################################
# tree illustration 
        
def estimate_position(self):
    """ 
    Estimate the x and y position in the plotting coordinates of each node in the tree, and add the x and y position to the data dictionary of each node.

    :param tree: the tree to estimate the position in
    :return: the tree with the x and y position added to the data dictionary of each node
    """
    
    # Determine Y coordinates, with distance from root 
    self.root.data["y_temp"] = 0
    for leaf in self.iter_dfs():
        # initlize an empty x coordinate for later 
        leaf.data["x_temp"] = 0
        if leaf.parent is not None: # Skip root
            if 'edge_length' in leaf.data.keys():
                leaf.data["y_temp"] = leaf.parent.data["y_temp"] -leaf.data["edge_length"] 

            else: 
                leaf.data["y_temp"] = leaf.parent.data["y_temp"] - 1
    # Define x coordinate for each leaf 
    for i,leaf in enumerate(self.iter_leaves_dfs()):
        leaf.data["x_temp"] = i


    # Determine X coordinates from bottom and up 
    for level in reversed(list(self.iter_levels())):
        for leaf in level:
                while leaf.parent is not None:
                    x_coordinate = 0
                    leaf = leaf.parent
                    for i,node in enumerate(leaf.children):
                        x_coordinate += node.data["x_temp"]
                    leaf.data["x_temp"] = x_coordinate/(i+1)
    return self 
     
def plot_tree_(self,ax=None,inc_names=False): 
    from matplotlib import pyplot as plt
    """
    Visualize the hierarchical structure of the tree

    :param tree: the tree to plot
    :param ax: the axis to plot the tree on, if None, a new figure is created, defaults to None
    :param inc_names: whether to include the names of the nodes in the plot, defaults to False
    """
 
    if ax == None:
        fig,ax = plt.subplots(figsize=(10,8))

    self = estimate_position(self)
    ax.plot(self.root.data["x_temp"], self.root.data["y_temp"], 'ko')  # Plot the current node
    ax.axis('off')
    for leaf in self.iter_bfs():
        if len(leaf.children) != 0:
            plot_node(leaf,ax,inc_names)

def plot_node(parent,ax,inc_names):
    """
    Plot a single node and its children

    :param parent: the parent node to plot
    :param ax: the axis to plot the node on
    :param inc_names: whether to include the names of the nodes in the plot
    """
    from matplotlib import pyplot as plt
    """Plot a single node and its children"""
    ax.plot(parent.data["x_temp"], parent.data["y_temp"], 'ko')  # Plot the current node
    if inc_names and parent.name is not None:
        ax.text(parent.data["x_temp"], parent.data["y_temp"], parent.name+" ", fontdict=None,rotation="vertical",va="top",ha="center")
        # Draw vertical line from parent to current level
    for child in parent.children:
        ax.plot(child.data["x_temp"], child.data["y_temp"], 'ko')

        # Include text 
        if inc_names and child.name is not None:
            ax.text(child.data["x_temp"], child.data["y_temp"], child.name+" ", fontdict=None,rotation="vertical",va="top",ha="center")
        # Draw vertical line from parent to current level
        ax.plot([child.data["x_temp"], parent.data["x_temp"]], [parent.data["y_temp"], parent.data["y_temp"]], 'k-')
        # Draw horizontal line to child
        ax.plot([child.data["x_temp"], child.data["x_temp"]], [parent.data["y_temp"], child.data["y_temp"]], 'k-') 


#####################################################################################################
# plot shape tree

def plot_tree_shape(self,ax=None,inc_names=False,shape="landmarks"): 
    from matplotlib import pyplot as plt
    """Plot the tree using matplotlib"""
 
    if ax == None:
        fig,ax = plt.subplots(figsize=(16,10))
        ax.axis('off')

    self = estimate_position_shape(self)
   

    n_leafs = len(list(self.iter_leaves()))
    scale = 7/8
    dis = 1/n_leafs*1/2*scale
    #print(dis)

    ####### DO all for root 
    leaf = self.root

    
    x = leaf.data["x_temp"]
    y =  leaf.data["y_temp"]
    
    # Include text
    if inc_names and leaf.name is not None:
        rotation = "horizontal" if len(leaf.name) < 3 else "vertical"
        ax.text(x, y-dis, leaf.name, fontdict=None, rotation=rotation, va="top", ha="center")


    points = scale_points(leaf.data[shape].reshape((-1,2)),[(x-dis,y-dis),(x+dis,y+dis)])
    for point in points:
        ax.plot(*point, 'ro')
        
    draw_box(ax, x, y, dis)

   # ax.axis('off')
    for leaf in self.iter_bfs():
        if len(leaf.children) != 0:
            plot_node_shape(leaf,ax,inc_names,dis,shape)

def estimate_position_shape(self):
    """ Estimate the x and y coordinates of each point """
    
    # Determine Y coordinates, with distance from root 
    self.root.data["y_temp"] = 0
    for leaf in self.iter_dfs():
        # initlize an empty x coordinate for later 
        leaf.data["x_temp"] = 0
        if leaf.parent is not None: # Skip root
            leaf.data["y_temp"] = leaf.parent.data["y_temp"] - leaf.data.get('edge_length', 1)

    # Define x coordinate for each leaf 
    for i,leaf in enumerate(self.iter_leaves_dfs()):
        leaf.data["x_temp"] = i


    # Determine X coordinates from bottom and up 
    for level in reversed(list(self.iter_levels())):
        for leaf in level:
                while leaf.parent is not None:
                    x_coordinate = 0
                    leaf = leaf.parent
                    for i,node in enumerate(leaf.children):
                        x_coordinate += node.data["x_temp"]
                    leaf.data["x_temp"] = x_coordinate/(i+1)


    
    # get the last leaf to see coordinates
    min_depth = 0 ;  min_width = 0
    # Get last leaf in leaves
    leaf = list(self.iter_leaves())[-1]
    max_depth = leaf.data["y_temp"]
    max_width = leaf.data["x_temp"]
    
    # NOrmalize everthing
    try:
        for leaf in self.iter_dfs():
            leaf.data["y_temp"] =- (leaf.data["y_temp"] - min_depth)/(max_depth - min_depth)
            leaf.data["x_temp"] = (leaf.data["x_temp"] - min_width)/(max_width - min_width)
    except ZeroDivisionError:
        pass

    #self.root.data["x_temp"] = 0.5
    return self 

def scale_points(points, bounding_box, padding=0.1):
    # Unpack bounding box coordinates
    box_min_x, box_min_y = bounding_box[0]
    box_max_x, box_max_y = bounding_box[1]

    # Calculate the range of the bounding box
    box_range_x = box_max_x - box_min_x
    box_range_y = box_max_y - box_min_y

    # Add padding to the bounding box
    box_min_x += box_range_x * padding
    box_max_x -= box_range_x * padding
    box_min_y += box_range_y * padding
    box_max_y -= box_range_y * padding
    box_range_x = box_max_x - box_min_x
    box_range_y = box_max_y - box_min_y

    # Find the min and max x, y in the points
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)

    # Calculate the range of the points
    range_x = max_x - min_x
    range_y = max_y - min_y

    # Scale the points to fit inside the bounding box
    scaled_points = []
    for x, y in points:
        # Avoid division by zero
        scaled_x = box_min_x + ((x - min_x) / range_x) * box_range_x
        scaled_y = box_min_y + ((y - min_y) / range_y) * box_range_y


        scaled_points.append((scaled_x, scaled_y))

    return scaled_points


def draw_box(ax, x, y, dis):
    #ax.plot([x-dis,x+dis],[y+dis,y+dis], 'k-')  # Upper Horizontal
    #ax.plot([x-dis,x+dis],[y-dis,y-dis], 'k-')  # Lower horizontal
    #ax.plot([x-dis,x-dis],[y-dis,y+dis], 'k-')  # Vertical lines
    #ax.plot([x+dis,x+dis],[y-dis,y+dis], 'k-')  # Vertical lines opposite
    ax.fill([x-dis, x+dis, x+dis, x-dis], 
            [y-dis, y-dis, y+dis, y+dis], color='white', edgecolor='black')
    


def plot_node_shape(parent, ax, inc_names, dis,shape):
    from matplotlib import pyplot as plt

    x0 = parent.data["x_temp"]
    y0 = parent.data["y_temp"]

    for child in parent.children:
        x = child.data["x_temp"]
        y = child.data["y_temp"]

        # Draw horizontal and vertical lines
        if len(parent.children) > 1:
            if x<x0-.5*dis or x>x0+.5*dis:
                ax.plot([x,x0-dis if x<x0 else x0+dis], [y0,y0],'k-')
            ax.plot([x,x],[y0 if x<x0-.5*dis or x>x0+.5*dis else y0-dis,y+dis],'k')      
        else:
            ax.plot([x,x],[y0-dis,y+dis],'k')

        # Draw box for shape
        draw_box(ax, x, y, dis)

        # Plot points
        points = scale_points(child.data[shape].reshape((-1,2)),[(x-dis,y-dis),(x+dis,y+dis)])
        for point in points:
            ax.plot(*point, 'ro')

        # Include text
        if inc_names and child.name is not None:
            rotation = "horizontal" if len(child.name) < 3 else "vertical"
            ax.text(x, y-dis, child.name, fontdict=None, rotation=rotation, va="top", ha="center")


######################################################################################################
# tree illustration in text format
# Functions for nicer tree printing in the terminal, modified from https://github.com/AharonSambol/PrettyPrintTree

def zip_longest(*iterables: list, default: Any) -> Generator:
    """
    Returns an iterator that aggregates elements from each of the input iterables.
    If the iterables are of unequal length, missing values are filled with the specified default value.

    :param default: the value to fill missing values with
    :yield: a tuple containing the elements from each of the input iterables
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

    :param text: the text to be adjusted.
    :param amount: the desired width of the adjusted text.
    :param padding: the character used for padding, defaults to ' '
    :return: the adjusted text
    """    
    return text + padding * (amount - len(text))

def right_adjust(text: str, amount: int, padding: str =' ') -> str:
    """
    Adjusts the given text to the right by adding padding characters.    

    :param text: the text to be adjusted.
    :param amount: the total width of the resulting string.
    :param padding: the character used for padding, defaults to ' '
    :return: the adjusted text.
    """   
    return padding * (amount - len(text)) + text

class TreeNodeFormatter:
    """
    A class that represents the formatter of a tree node.
    """

    @classmethod
    def from_string(cls: Any, string: str) -> Any:
        """
        Create a TreeNodeFormatter object from a string representation.

        :param cls: the TreeNodeFormatter class.
        :param string: the string representation of the tree node.
        :return: the TreeNodeFormatter object
        """        
        lines = string.split('\n')
        height = len(lines)
        width = max(len(line) for line in lines)
        return cls(lines, height=height, width=width)
    
    def __init__(self, lines: list[str], *, height: int, width = int, middle_height: int = None, middle_width: int = None) -> None:
        """
        Initialize a TreeNodeFormatter object.        

        :param lines: the content of the tree node.
        :param height: the height of the tree node text.
        :param width: the width of the tree node text.
        :param middle_height: the middle height position of the tree node text, defaults to None
        :param middle_width: the middle width position of the tree node text, defaults to None
        """        
        self.lines = lines
        self.height = height
        self.width = width
        self._middle_height = middle_height
        self._middle_width = middle_width

    def to_string(self) -> str:
        """
        Convert the TreeNodeFormatter object to a string representation.

        :return: the string representation of the tree node.
        """        
        return '\n'.join(self.lines)
    
    @property
    def middle_width(self) -> int:
        """
        Get the middle width of the tree node text.

        :return: the middle width position of the tree node text.
        """        
        if self._middle_width is None:
            return sum(divmod(self.width, 2)) - 1
        return self._middle_width
    
    @property
    def middle_height(self) -> int:
        """ 
        Get the middle height of the tree node text.

        :return: _description_
        """        
        if self._middle_height is None:
            return sum(divmod(self.height, 2)) - 1
        return self._middle_height

def add_parent(parent_formatter: TreeNodeFormatter, children_formatter: TreeNodeFormatter) -> TreeNodeFormatter:
    """
    Combines the formatting information of a parent node and its children nodes to create a new TreeNodeFormatter object.    

    :param parent_formatter: the formatting information of the parent node.
    :param children_formatter: the formatting information of the children nodes.
    :return: a new TreeNodeFormatter object with the combined formatting information.
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
    """
    def __init__(self, tree: HypTree) -> None:
        """
        Initializes the HypTreeFormatter object with the given HypTree object.

        :param tree: the HypTree object to format.
        """        
        self.root = tree.root

    def get_children(self, node: TreeNode) -> list[TreeNode]:
        """
        Get the children of the given node.

        :param node: the node to get the children of.
        :return: the children of the given node.
        """        
        return node.children
    
    def get_name(self, node: TreeNode) -> str:
        """
        Get the name of the given node.

        :param node: the node to get the name of.
        :return: the name of the given node, return '*' if the name is None.
        """        
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

        :return: the formatted tree structure as a string.
        """
        res = self.tree_join_formatter(self.root)
        res = res.to_string().rstrip()
        return res
    
    def tree_join_formatter(self, node: TreeNode, depth: int = 0) -> TreeNodeFormatter:
        """
        Recursively formats the tree structure and returns the root node formatter.

        :param node: the current node being formatted.
        :param depth: the depth of the current node in the tree structure.
        :return: the root node formatter.
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

        :param node: the node to convert.
        :return: the converted node formatter.
        """
        name = self.get_name(node)
        return TreeNodeFormatter.from_string(str(name))
    
    def print_tree(self) -> None:
        """
        Prints the formatted tree representation to the console.
        """
        print(self.format())

def join_horizontally(boxes: list[TreeNodeFormatter]) -> TreeNodeFormatter:
    """
    Joins multiple TreeNodeFormatter boxes horizontally to form a line to print.

    :params boxes: a list of TreeNodeFormatter objects to gather.:
    :return: the joined TreeNodeFormatter box.
    """
    lines, width, height = join_boxes(boxes)
    middle = add_pipes(boxes, lines)
    height += 1
    return TreeNodeFormatter(lines, height=height, width=width, middle_width=middle)

def join_boxes(boxes: list[TreeNodeFormatter]) -> tuple[list[str], int, int]:
    """
    Joins multiple TreeNodeFormatter together horizontally to get the text, width and height of this line.

    :param boxes: a list of TreeNodeFormatter objects to gather.
    :return: a tuple containing the joined lines, width, and height of the joined boxes.
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

def add_pipes(boxes: list[TreeNodeFormatter], lines: list[str]) -> int:
    """
    Adds pipes to the given lines to create a tree-like structure.

    :param boxes: a list of TreeNodeFormatter objects to gather.
    :param lines: a list of strings representing the lines of the tree.
    :return: the number of characters added as padding to the lines.
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

      
