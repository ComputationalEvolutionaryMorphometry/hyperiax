from typing import Type

from .tree import TopologyNode, HypTree

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


### Alternative Newick tree generation
def tree_from_newick_recursive(newick_str: str) -> HypTree:
    """ 
    Generate a tree from a Newick string recursively.

    :param newick_str: newick string representation
    :return: The constructed tree
    """
    import re

    iter_tokens = re.finditer(r"([^:;,()\s]*)(?:\s*:\s*([\d.]+)\s*)?([,);])|(\S)", newick_str+";")

    def recursive_parse_newick(parent=None):
        name, length, delim, char = next(iter_tokens).groups(0)

        node = TopologyNode(name=name if name else None,             # create a "ghost" subtree root node without data
                       
                       parent=parent,
                       children=[])
        node.data = {"edge_length": float(length)} if length else {}

        
        if char == "(": # start a subtree

            while char in "(,": # add all children within a parenthesis to the current node
                child, char = recursive_parse_newick(parent=node)
                node.children.append(child)

            name, length, delim, char = next(iter_tokens).groups(0)

            node.name = name                        # assign data to the "ghost" subtree root node
            node.data = {"edge_length": float(length)} if length else {}
            
        return node, delim
    
    root, _ = recursive_parse_newick()

    tree = HypTree(root)

    if 'edge_length' in root.data.keys():
        tree.add_property('edge_length', (1,), dtype=float)

        for node in tree.iter_topology_bfs():
            tree.data['edge_length'] = tree.data['edge_length'].at[node.id].set(node.data['edge_length'])
            del node.data
    
    return tree
