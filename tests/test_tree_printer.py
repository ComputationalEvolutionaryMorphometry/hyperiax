from hyperiax.tree.builders import symmetric_tree
from hyperiax.tree.printer_utils import HypTreeFormatter

def test_tree_formatter():
    tree = symmetric_tree(h=4, degree=3)
    tree.plot_tree_text()