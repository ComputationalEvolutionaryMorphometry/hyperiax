from typing import Type

from .tree import TopologyNode

def symmetric_topology(height : int, degree : int, fake_root : bool = False) -> TopologyNode:
    """ Generate tree of given height and degree

    A tree of height zero contains just the root;
    a tree of height one contains the root and one level of leaves below it,
    and so forth.

    :param height: The height of the tree
    :param degree: The degree of each node in the tree
    :param new_node: The node used to construct the tree, defaults to TreeNode
    :param fake_root: The fake root node, defaults to None
    :raises ValueError: If height is negative
    :return: The constructed tree
    """
    assert height >= 0, f'Height expected to be nonngeative, received {height=}.'

    def _builder(h: int, degree: int, parent):
        node = TopologyNode(); node.parent = parent; node.children = []
        if h > 1:
            node.children = [_builder(h - 1, degree, node) for _ in range(degree)]
        return node

    return _builder(height + 1, degree, None)



def asymmetric_topology(h: int)  -> TopologyNode:
    """ 
    Generate an asymmetric binary tree of given height.
    A tree of height zero contains just the root;
    a tree of height one contains the root and one level of leaves below it, and so forth.

    :param h: The height of the tree
    :raises ValueError: If height is negative
    :return: The constructed tree
    """
    if h < 0:
        raise ValueError(f'Height shall be nonnegative integer, received {h=}.')
    elif h == 0:
        return ChildList([TreeNode()])

    # Fake root 
    root = TopologyNode(); root.parent = None; root.children = []
    root.children = [TopologyNode(children= [],parent=root), TopologyNode(children=[],parent=root)]
   
    node = root.children[0]

    for _ in range(h - 1):
        node.children=  [TopologyNode(children=[],parent=node), TopologyNode(children=[],parent=node)]

        node = node.children[0]

    return root 


