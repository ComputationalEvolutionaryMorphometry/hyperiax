from hyperiax.tree.builders import symmetric_tree, tree_from_newick_recursive, tree_from_newick
from hyperiax.tree.printer_utils import HypTreeFormatter

def test_tree_formatter():
    tree = symmetric_tree(h=4, degree=3)
    tree.plot_tree_text()


def test_tree_from_newick_recursive():
    newick_str = "(A,B,(C,D)E)F;"
    tree = tree_from_newick_recursive(newick_str)
    tree.plot_tree_text()
    tree = tree_from_newick(newick_str)
    tree.plot_tree_text()