from ..tree import HypTree

from matplotlib import pyplot as plt
from matplotlib import patches as mpatch

cmap = lambda x: plt.cm.ocean(x*.9) # use a slightly darker color for the levels

def plot_tree_2d_scatter(tree : HypTree, property : str, ax=None, selector=lambda z: z):

    if ax == None:
        fig,ax = plt.subplots(figsize=(10,8))

    for i, (l_start, l_end) in enumerate(tree.levels):
        for nodeidx in range(l_start, l_end):
            cdat = selector(tree.data[property][nodeidx])
            cdat = cdat if len(cdat.shape) <= 2 else cdat[-1] # possibly to last value if dat is a trajectory of values
            if len(cdat.shape) == 1:
                ax.scatter(*cdat, color=cmap(i/len(tree.levels)))
            else:
                ax.scatter(cdat[:,0], cdat[:,1], color=cmap(i/len(tree.levels)))

            # plot transition from parent to child
            if nodeidx == 0: continue
            parentidx = tree.parents[nodeidx]
            dat = selector(tree.data[property][parentidx])
            dat = dat if len(dat.shape) <= 2 else dat[-1] # possibly to last value if dat is a trajectory of values
            cdat = selector(tree.data[property][nodeidx])
            if len(cdat.shape) == 1:
                ax.arrow(*dat, *(cdat-dat), width=0.01, length_includes_head=True, color=cmap(i/len(tree.levels)))
            elif len(cdat.shape) == 2:
                for j in range(cdat.shape[0]):
                    ax.plot([dat[j,0],cdat[j,0]], [dat[j,1],cdat[j,1]], color=cmap(i/len(tree.levels)))
            elif len(cdat.shape) == 3:
                for j in range(cdat.shape[1]):
                    ax.plot(cdat[:,j,0], cdat[:,j,1], color=cmap(i/len(tree.levels)))

        handles = [mpatch.Patch(color=cmap(i/len(tree.levels)), label = f'{i+1}') for i in range(len(tree.levels))]
        legend = ax.legend(handles=handles, title="Levels")
        ax.add_artist(legend)
        ax.grid(True)

def plot_tree_3d_scatter(tree : HypTree, property : str, ax=None, selector=lambda z: z):

    if ax == None:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
    levels = list(tree.iter_topology_levels())

    for i, level in enumerate(levels):
        for node in level:
            dat = selector(tree.data[property][node.id])
            dat = dat if len(dat.shape) == 1 else dat[-1] # possibly to last value if dat is a trajectory of values
            if len(dat.shape) == 1:
                ax.scatter(*dat, color=cmap(i/len(tree.levels)))
            else:
                ax.scatter(dat[:,0], dat[:,1], dat[:,2], color=cmap(i/len(tree.levels)))
            if 'name' in tree.data.keys():
                ax.text(*dat, tree.data['name'][node.id], color='black')

            # plot transition from parent to child
            if node.children:
                for child in node.children:
                    cdat = selector(tree.data[property][child.id])
                    if len(cdat.shape) == 1:
                        ax.quiver(*dat, *(cdat-dat), length=1.0, arrow_length_ratio=0.1, color=cmap(i/len(tree.levels)))
                    elif len(cdat.shape) == 2:
                        ax.plot(cdat[:,0], cdat[:,1], cdat[:,2], color=cmap(i/len(tree.levels)))
                    elif len(cdat.shape) == 3:
                        for j in range(cdat.shape[1]):
                            ax.plot(cdat[:,j,0], cdat[:,j,1], cdat[:,j,2], color=cmap(i/len(tree.levels)))

    handles = [mpatch.Patch(color=cmap(i/len(levels)), label=f'{i+1}') for i in range(len(levels))]
    legend = ax.legend(handles=handles, title="Levels")
    ax.add_artist(legend)
    ax.grid(True)