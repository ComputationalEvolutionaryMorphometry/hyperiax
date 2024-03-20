from hyperiax.tree.builders import THeight_legacy
from hyperiax.tree.printer_utils import HypTreeFormatter

def test_tree_formatter():
    tree = THeight_legacy(h=4, degree=3)
    tree.plot_tree_text()