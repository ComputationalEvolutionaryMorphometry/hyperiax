# Functions for plotting data and tree
import numpy as np

#####################################################################################################
# 2d plot of data points in a 

def plot_tree_2d_(self, ax=None, selector=None):
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
    """ Estimate the x and y coordinates of each point """
    
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
    """Plot the tree using matplotlib"""
 
    if ax == None:
        fig,ax = plt.subplots(figsize=(10,8))

    self = estimate_position(self)
    ax.plot(self.root.data["x_temp"], self.root.data["y_temp"], 'ko')  # Plot the current node
    ax.axis('off')
    for leaf in self.iter_bfs():
        if len(leaf.children) != 0:
            plot_node(leaf,ax,inc_names)

def plot_node(parent,ax,inc_names):
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



      
