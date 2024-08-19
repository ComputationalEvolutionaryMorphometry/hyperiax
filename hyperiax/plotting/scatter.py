from ..tree import HypTree

def plot_tree_2d_scatter(tree : HypTree, property : str, ax=None):
    from matplotlib import pyplot as plt
    from matplotlib import patches as mpatch


    cmap = plt.cm.ocean

    if ax == None:
        fig,ax = plt.subplots(figsize=(10,8))

    for i in range(1, len(tree)):
        cdat = tree.data[property][i]
        if i == 0: continue
        parentidx = tree.parents[i]
        dat = tree.data[property][parentidx]
        ax.arrow(*dat, *(cdat-dat), width=0.01, length_includes_head=True, color='gray')

    for i, (l_start, l_end) in enumerate(tree.levels):
        for nodeidx in range(l_start, l_end):
            cdat = tree.data[property][nodeidx]
            ax.scatter(*cdat, color=cmap(i/len(tree.levels)))

        handles = [mpatch.Patch(color=cmap(i/len(tree.levels)), label = f'{i+1}') for i in range(len(tree.levels))]
        legend = ax.legend(handles=handles, title="Levels")
        ax.add_artist(legend)
        ax.grid(True)
