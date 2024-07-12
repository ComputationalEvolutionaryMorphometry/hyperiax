from ..tree import HypTree

from jax import numpy as jnp
import copy

        
def estimate_position(self):
    """ 
    Estimate the x and y position in the plotting coordinates of each node in the tree, and add the x and y position to the data dictionary of each node.

    :param tree: the tree to estimate the position in
    :return: the tree with the x and y position added to the data dictionary of each node
    """
    
    # Determine Y coordinates, with distance from.topology_root 
    depth = jnp.max(self.node_depths)
    self.data['y_temp'] = self.data['y_temp'].at[0].set(depth)
    for leaf in self.iter_topology_dfs():
        # initlize an empty x coordinate for later 
        self.data['x_temp'] = self.data['x_temp'].at[leaf.id].set(0)

        if leaf.parent is not None: # Skip.topology_root
            if 'edge_length' in self.data.keys():
                new_value = self.data['y_temp'][leaf.parent.id]-self.data['edge_length'][leaf.id]
                self.data['y_temp'] = self.data['y_temp'].at[leaf.id].set(new_value)

            else: 
                depth = jnp.max(self.node_depths)-self.node_depths[leaf.id]
                self.data['y_temp'] = self.data['y_temp'].at[leaf.id].set(depth)

    # Define x coordinate for each leaf 
    for i,leaf in enumerate(self.iter_topology_leaves_dfs()):
        self.data['x_temp'] = self.data['x_temp'].at[leaf.id].set(i)


    # Determine X coordinates from bottom and up 
    for level in reversed(list(self.iter_topology_levels())):
        for leaf in level:
                while leaf.parent is not None:
                    x_coordinate = 0
                    leaf = leaf.parent
                    for i,node in enumerate(leaf.children):
                        x_coordinate += self.data['x_temp'][node.id]
                    self.data['x_temp'] = self.data['x_temp'].at[leaf.id].set(x_coordinate/(i+1))

     
def plot_tree(self_input,ax=None,inc_names=False): 
    from matplotlib import pyplot as plt
    """
    Visualize the hierarchical structure of the tree

    :param tree: the tree to plot
    :param ax: the axis to plot the tree on, if None, a new figure is created, defaults to None
    :param inc_names: whether to include the names of the nodes in the plot, defaults to False
    """

    self = copy.deepcopy(self_input)

    self.add_property('x_temp', shape=(1,))
    self.add_property('y_temp', shape=(1,))
 
    if ax == None:
        fig,ax = plt.subplots(figsize=(10,8))

    estimate_position(self)

   
    ax.plot(self.data["x_temp"], self.data["y_temp"], 'ko') # plot all nodes
    ax.axis('off')
     
    # Plot the connection for each node 
    for leaf in self.iter_topology_bfs():
        if len(leaf.children) != 0:
           # 1+1
            plot_node(self,leaf,ax,inc_names)

def plot_node(self,parent,ax,inc_names):
    """
    Plot a single node and its children

    :param parent: the parent node to plot
    :param ax: the axis to plot the node on
    :param inc_names: whether to include the names of the nodes in the plot
    """
    from matplotlib import pyplot as plt
    """Plot a single node and its children"""
    ax.plot(self.data["x_temp"][parent.id],self.data["y_temp"][parent.id], 'ko')  # Plot the current node

    if inc_names:
        ax.text(self.data["x_temp"][parent.id].item(),self.data["y_temp"][parent.id].item(),  parent.name+"  ", fontdict=None,rotation="vertical",va="top",ha="center")
        1+1
    for child in parent.children:
        if inc_names:
            ax.text(self.data["x_temp"][child.id].item(),self.data["y_temp"][child.id].item(), child.name+"  ", fontdict=None,rotation="vertical",va="top",ha="center")
        
        
        # Draw vertical line to parent
        ax.plot([self.data["x_temp"][child.id],self.data["x_temp"][parent.id]], 
                [self.data["y_temp"][parent.id],self.data["y_temp"][parent.id]], 'k-')
        # Draw horizontal line to child
        ax.plot([self.data["x_temp"][child.id], self.data["x_temp"][child.id]], 
                [self.data["y_temp"][parent.id], self.data["y_temp"][child.id]], 'k-') 



