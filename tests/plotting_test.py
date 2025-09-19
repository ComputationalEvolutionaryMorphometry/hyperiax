from hyperiax.tree import TopologyNode, HypTree, topology
from hyperiax.plotting.ascii_new import plot_tree


def test_plot_tree():
    topo = topology.asymmetric_topology(3)
    tree = HypTree(topo)

    print("\nVertical plot:")
    plot_tree(tree, horizontal=False)

    print("\nHorizontal plot:")
    plot_tree(tree, horizontal=True)
