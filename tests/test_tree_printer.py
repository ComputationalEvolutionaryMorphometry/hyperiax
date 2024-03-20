from hyperiax.tree.builders import THeight_legacy
from hyperiax.tree.utils import HypTreeFormatter

def test_tree_formatter():
    tree = THeight_legacy(h=5, degree=3)
    tree_formatter = HypTreeFormatter(tree)
    tree_formatter.print_tree()