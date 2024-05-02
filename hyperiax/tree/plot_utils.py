# Functions for plotting data and tree


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

        for child in parent.children:
            ax.plot(child.data["x_temp"], child.data["y_temp"], 'ko')

            # Include text 
            if inc_names:
                if child.name is not None:
                    ax.text(child.data["x_temp"], child.data["y_temp"], child.name+" ", fontdict=None,rotation="vertical",va="top",ha="center")
            # Draw vertical line from parent to current level
            ax.plot([child.data["x_temp"], parent.data["x_temp"]], [parent.data["y_temp"], parent.data["y_temp"]], 'k-')
            # Draw horizontal line to child
            ax.plot([child.data["x_temp"], child.data["x_temp"]], [parent.data["y_temp"], child.data["y_temp"]], 'k-') 


