from ..tree import HypTree
from jax import numpy as jnp
import copy

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import patches as mpatch
import plotly.graph_objects as go
        
def estimate_position(self,normalize=True):
    """ 
    Estimate the x and y position in the plotting coordinates of each node in the tree, and add the x and y position to the data dictionary of each node.

    :param tree: the tree to estimate the position in
    :return: the tree with the x and y position added to the data dictionary of each node
    """
    
    # Determine Y coordinates, with distance from.topology_root 
    depth = jnp.max(self.node_depths)
    self.data['y_temp'] = self.data['y_temp'].at[self.topology_root.id].set(0)
    for leaf in self.iter_topology_dfs():
        # initlize an empty x coordinate for later 
        self.data['x_temp'] = self.data['x_temp'].at[leaf.id].set(0)

        if leaf.parent is not None: # Skip.topology_root
            if 'edge_length' in self.data.keys():
                new_value = self.data['y_temp'][leaf.parent.id]-self.data['edge_length'][leaf.id]
                self.data['y_temp'] = self.data['y_temp'].at[leaf.id].set(new_value)
            else: 
                self.data['y_temp'] = self.data['y_temp'].at[leaf.id].set(-self.node_depths[leaf.id])

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
    
    if normalize:
        # get the last leaf to see coordinates
        min_depth = 0 ;  min_width = 0
        # Get last leaf in leaves
        leaf = list(self.iter_topology_leaves_bfs())[-1]
        max_depth = self.data["y_temp"][leaf.id]
        max_width = self.data["x_temp"][leaf.id]
    
        # Normalize everthing
        try:
            for leaf in self.iter_topology_dfs():
                self.data['y_temp'] = self.data['y_temp'].at[leaf.id].set(-(self.data["y_temp"][leaf.id] - min_depth)/(max_depth - min_depth))
                self.data['x_temp'] = self.data['x_temp'].at[leaf.id].set((self.data["x_temp"][leaf.id] - min_width)/(max_width - min_width))
        except ZeroDivisionError:
            pass

     
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
    estimate_position(self)
   
    if ax == None:
        fig,ax = plt.subplots(figsize=(10,8))

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

    if inc_names and parent.name is not None and not parent.name.startswith('xx_'):
        ax.text(self.data["x_temp"][parent.id].item(),self.data["y_temp"][parent.id].item(),  parent.name+"  ", fontdict=None,rotation="vertical",va="top",ha="center")
 

    for child in parent.children:
        if inc_names and child.name is not None and not child.name.startswith('xx_'):
            ax.text(self.data["x_temp"][child.id].item(),self.data["y_temp"][child.id].item(), child.name+"  ", fontdict=None,rotation="vertical",va="top",ha="center")
        
        
        # Draw vertical line to parent
        ax.plot([self.data["x_temp"][child.id],self.data["x_temp"][parent.id]], 
                [self.data["y_temp"][parent.id],self.data["y_temp"][parent.id]], 'k-')
        # Draw horizontal line to child
        ax.plot([self.data["x_temp"][child.id], self.data["x_temp"][child.id]], 
                [self.data["y_temp"][parent.id], self.data["y_temp"][child.id]], 'k-') 



#####################################################################################################
# plot shape tree

def plot_tree_2d_shape(self_input : HypTree, property : str,ax=None,inc_names=False): 
    from matplotlib import pyplot as plt
    """Plot the tree using matplotlib"""

    self = copy.deepcopy(self_input)
    self.add_property('x_temp', shape=(1,))
    self.add_property('y_temp', shape=(1,))
    estimate_position(self,normalize=True)
 
    if ax == None:
        fig,ax = plt.subplots(figsize=(16,10))
        ax.axis('off')

    n_leafs = len(list(self.iter_topology_leaves_bfs()))
    scale = 7/8
    dis = 1/n_leafs*1/2*scale

    ####### DO all for root 
    leaf = self.topology_root

    x = self.data["x_temp"][leaf.id]
    y = self.data["y_temp"][leaf.id]
    
    # Include text
    if inc_names and leaf.name is not None:
        rotation = "horizontal" if len(leaf.name) < 3 else "vertical"
        ax.text(x, y-dis, leaf.name, fontdict=None, rotation=rotation, va="top", ha="center")

    plot_trajectory = len(self.data[property][self.topology_root.id].shape) > 1
    if not plot_trajectory:
        points = scale_points(self.data[property][leaf.id].reshape((-1,2)),[(x-dis,y-dis),(x+dis,y+dis)])
    else:
        points = scale_points(self.data[property][leaf.id][-1].reshape((-1,2)),[(x-dis,y-dis),(x+dis,y+dis)])
    for point in points:
        ax.plot(*point, 'r.')
    self.add_property('temp_plotted_point', shape=np.array(points).shape)
    self.data['temp_plotted_point'] = self.data['temp_plotted_point'].at[self.topology_root.id].set(points)
        
    plot_trajectory = len(self.data[property][leaf.id].shape) > 1
    if not plot_trajectory:
        draw_box(ax, x, y, dis) # regular case (no trajectory)

    # ax.axis('off')
    n_levels = len(self.levels)
    for i, level in enumerate(self.iter_topology_levels()):
        for node in level:
            if len(node.children) != 0:
                plot_node_shape(self,node,ax,inc_names,dis,property,i/n_levels)

    cmap = plt.cm.ocean
    handles = [mpatch.Patch(color=cmap(i/n_levels), label = f'{i+1}') for i in range(n_levels)]
    legend = ax.legend(handles=handles, title="Levels")
    ax.add_artist(legend)

def plot_tree_3d_shape(self_input : HypTree,property : str,fig=None,inc_names=False,scale=1.,mesh=None): 
    import plotly.graph_objects as go
    """Plot the tree using plotly"""

    self = copy.deepcopy(self_input)
    self.add_property('p_temp', shape=(3,))
 
    if fig == None:
        fig = go.Figure()
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False), 
                yaxis=dict(visible=False), 
                zaxis=dict(visible=False)
            ),
            showlegend=False,  # Disable legends
            #width = 1000,
            height = 800,
        )

    # simple placement in xz plane, does not take edge lengths into account
    x_span = 1.5*len(self.is_leaf)
    n_levels = len(self.levels)
    level_z = 2.
    def set_pos(node,childi,nchildren,x_span):
        if node.parent is not None:
            x = self.data['p_temp'][node.parent.id][0] + (childi-(nchildren-1)/2)*x_span
            y = 0.
            z = self.data['p_temp'][node.parent.id][2] - level_z
        else:
            x = 0.; y = 0.; z = 0.
        self.data['p_temp'] = self.data['p_temp'].at[node.id].set(np.array([x,y,z]))
        for i,child in enumerate(node.children):
            set_pos(child,i,len(node.children),x_span/len(node.children))
    set_pos(self.topology_root,0,len(self.topology_root.children),x_span)

    ####### DO all for root 
    plot_trajectory = len(self.data[property][self.topology_root.id].shape) > 1
    if not plot_trajectory:
        points = scale*self.data[property][self.topology_root.id].reshape((-1,3))
    else:
        points = scale*self.data[property][self.topology_root.id][-1].reshape((-1,3))
    if mesh is None:
        fig.add_trace(go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers', marker=dict(color='blue')))
    else:
        plot_mesh_plotly(mesh,points,fig)
    self.add_property('temp_plotted_point', shape=points.shape)
    self.data['temp_plotted_point'] = self.data['temp_plotted_point'].at[self.topology_root.id].set(points)

    n_levels = len(self.levels)
    for i, level in enumerate(self.iter_topology_levels()):
        for node in level:
            if len(node.children) != 0:
                plot_node_shape_3d(self,node,fig,property,scale,i/n_levels,mesh=mesh)
    fig.update_layout(scene_aspectmode='data')
    return fig

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
    points=np.array(points)
    min_x,min_y=np.min(points,axis=0)
    max_x,max_y=np.max(points,axis=0)

    # Calculate the range of the points
    range_x = max_x - min_x
    range_y = max_y - min_y

    # Scale the points to fit inside the bounding box
    min_x,min_y = np.min(points,axis=0)
    max_x,max_y = np.max(points,axis=0)
    range_x = max_x-min_x
    range_y = max_y-min_y

    # Avoid division by zero
    scaled_x = box_min_x+((points[:,0]-min_x)/range_x)*box_range_x
    scaled_y = box_min_y+((points[:,1]-min_y)/range_y)*box_range_y

    return np.column_stack((scaled_x,scaled_y))


def draw_box(ax, x, y, dis):
    #ax.plot([x-dis,x+dis],[y+dis,y+dis], 'k-')  # Upper Horizontal
    #ax.plot([x-dis,x+dis],[y-dis,y-dis], 'k-')  # Lower horizontal
    #ax.plot([x-dis,x-dis],[y-dis,y+dis], 'k-')  # Vertical lines
    #ax.plot([x+dis,x+dis],[y-dis,y+dis], 'k-')  # Vertical lines opposite
    ax.fill([x-dis, x+dis, x+dis, x-dis], 
            [y-dis, y-dis, y+dis, y+dis], color='white', edgecolor='black')
    


def plot_node_shape(self, parent, ax, inc_names, dis, property, level, zorder_traj=1, zorder_box=2):
    from matplotlib import pyplot as plt

    x0 = self.data["x_temp"][parent.id]
    y0 = self.data["y_temp"][parent.id]

    cmap = plt.cm.ocean

    for child in parent.children:
        x = self.data["x_temp"][child.id]
        y = self.data["y_temp"][child.id]

        # plot just point configuration of entire trajectory
        plot_trajectory = len(self.data[property][child.id].shape) > 1

        if not plot_trajectory:
            # Draw horizontal and vertical lines with lower zorder
            if len(parent.children) > 1:
                if x<x0-.5*dis or x>x0+.5*dis:
                    ax.plot([x,x0-dis if x<x0 else x0+dis], [y0,y0],'k-', zorder=zorder_traj)
                ax.plot([x,x],[y0 if x<x0-.5*dis or x>x0+.5*dis else y0-dis,y+dis],'k', zorder=zorder_traj)      
            else:
                ax.plot([x,x],[y0-dis,y+dis],'k', zorder=zorder_traj)

            # Plot points with lower zorder
            points = scale_points(self.data[property][child.id].reshape((-1,2)),[(x-dis,y-dis),(x+dis,y+dis)])
            ax.scatter(points[:,0], points[:,1],color='r',marker='.', zorder=zorder_traj)
            
            # Draw box with higher zorder
            draw_box(ax, x, y, dis, zorder=zorder_box)

            self.data['temp_plotted_point'] = self.data['temp_plotted_point'].at[child.id].set(points)
        else:
            # Plot trajectory with lower zorder
            points = scale_points(self.data[property][child.id].reshape((-1,2)),[(x-dis,y-dis),(x+dis,y+dis)]).reshape((self.data[property][child.id].shape[0],-1,2))
            if 'temp_plotted_point' in self.data.keys():
                # linearly interpolate
                child_first_point = points[0]
                child_last_point = points[-1]
                parent_last_point = self.data['temp_plotted_point'][parent.id]
                num_points = self.data[property][child.id].shape[0]
                interpolated_array = np.linspace(parent_last_point-child_first_point, np.zeros_like(child_last_point), num_points)
                points = points+interpolated_array
            for i in range(points.shape[1]):
                ax.plot(points[:,i,0], points[:,i,1],color=cmap(level), zorder=zorder_traj)
                ax.plot(*points[-1,i], 'r.', zorder=zorder_box)
            self.data['temp_plotted_point'] = self.data['temp_plotted_point'].at[child.id].set(points[-1])

        # Include text
        if inc_names and child.name is not None:
            rotation = "horizontal" if len(child.name) < 3 else "vertical"
            ax.text(x, y-dis, child.name, fontdict=None, rotation=rotation, va="top", ha="center")


def plot_node_shape_3d(self : HypTree, parent,fig,shape,scale,level,mesh=None):
    cmap = plt.cm.ocean

    for child in parent.children:
        p = self.data['p_temp'][child.id]

        # plot just point configuration of entire trajectory
        plot_trajectory = len(self.data[shape][child.id].shape) > 1

        if not plot_trajectory:
            # no trajectory
            point = scale*self.data[shape][child.id].reshape((-1,3)) + p[None,:]
            if 'temp_plotted_point' in self.data.keys():
                points = np.vstack((self.data['temp_plotted_point'][parent.id],point)).reshape([2,-1,3])
            else:
                points = point.reshape([1,-1,3])
        else:
            # trajectory
            points = scale*self.data[shape][child.id].reshape((self.data[shape][child.id].shape[0],-1,3)) + p[None,None,:]
            if 'temp_plotted_point' in self.data.keys():
                # linearly interpolate
                child_first_point = points[0]
                child_last_point = points[-1]
                parent_last_point = self.data['temp_plotted_point'][parent.id]
                num_points = self.data[shape][child.id].shape[0]
                interpolated_array = np.linspace(parent_last_point-child_first_point, np.zeros_like(child_last_point), num_points)
                points = points+interpolated_array

        for i in range(points.shape[1]):
            fig.add_trace(go.Scatter3d(x=points[:,i,0], y=points[:,i,1], z=points[:,i,2], mode='lines', line=dict(color=cmap(level))))
            if mesh is None:
                fig.add_trace(go.Scatter3d(x=[points[-1,i,0]], y=[points[-1,i,1]], z=[points[-1,i,2]], mode='markers', marker=dict(color='red', size=6)))
        if mesh is not None:
            plot_mesh_plotly(mesh,points[-1],fig,edgecolor='red')
        self.data['temp_plotted_point'] = self.data['temp_plotted_point'].at[child.id].set(points[-1])

def plot_mesh_plotly(mesh,points,fig,color='lightblue',edgecolor='blue'):
    fig.add_trace(go.Mesh3d(x=points[:,0],y=points[:,1],z=points[:,2],i=mesh.faces[:,0],j=mesh.faces[:,1],k=mesh.faces[:,2],color=color,opacity=0.95))
    edges = np.vstack([mesh.faces[:,[0,1]],mesh.faces[:,[1,2]],mesh.faces[:,[2,0]]])
    x_edges = np.hstack([points[edges[:,0],0],points[edges[:,1],0],np.full(edges.shape[0],None)])
    y_edges = np.hstack([points[edges[:,0],1],points[edges[:,1],1],np.full(edges.shape[0],None)])
    z_edges = np.hstack([points[edges[:,0],2],points[edges[:,1],2],np.full(edges.shape[0],None)])
    fig.add_trace(go.Scatter3d(x=x_edges,y=y_edges,z=z_edges,mode='lines',line=dict(color=edgecolor,width=2)))