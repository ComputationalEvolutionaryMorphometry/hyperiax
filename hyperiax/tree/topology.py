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

def read_topology(newick_str: str, return_topology=False, precompute_child_gathers=True) -> HypTree:
    """ 
    Generate a tree from a Newick string recursively.

    :param newick_str: newick string representation
    :return: The constructed tree
    """
    import re

    iter_tokens = re.finditer(r"([A-Za-z0-9_-]+)?(?:\s*:\s*([\d.]+))?([,();])", newick_str)
    edge_lengths = []

    def recursive_parse_newick(parent=None):
        name, length, char = next(iter_tokens).groups()
        node = TopologyNode(name=name if name else "", parent=parent, children=[])
        if length:
            edge_lengths.append(length)

        if char == "(":
            while char in "(,":
                child, char = recursive_parse_newick(parent=node)
                if child is not None:
                    node.children.append(child)
   
            name, length, char = next(iter_tokens).groups()
            if length:
                edge_lengths.append(length)
        return node, char


    root, _ = recursive_parse_newick()

    if return_topology:
        return root
    else: 

        tree = HypTree(root,precompute_child_gathers)
        if root.children:
            tree.add_property('edge_length', (1,), dtype=float)

            for temp_length,node in zip(edge_lengths,list(reversed(list(tree.iter_topology_dfs())))):
                tree.data["edge_length"] = tree.data["edge_length"].at[node.id].set(float(temp_length))
                node.data = {"edge_length": float(temp_length)} if temp_length else {} # Should consider to be removed, but is used for plotting atm.


        return tree

def write_topology(tree:HypTree) -> str:
    """
    Convert a tree to a Newick string representation.

    :param tree: The tree to convert
    :return: Newick string representation of the tree
    """
    # Check if edge_length exists, otherwise fill it with ones
    if "edge_length" not in tree.data:
        tree.add_property('edge_length', shape=(1,))
        tree.data['edge_length'] =  tree.data['edge_length'].at[:].set(1)

    def recursive_to_newick(node) -> str:
        edge_length = tree.data["edge_length"].at[node.id].get().item()

        if not node.children:
            return f"{node.name}:{edge_length:.4f}" if node.name else f":{edge_length:.4f}"
        
        children_str = ",".join(recursive_to_newick(child) for child in node.children)
        node_str = f"({children_str}){node.name}:{edge_length:.4f}" if node.name else f"({children_str}):{edge_length:.4f}"
        return node_str

    return recursive_to_newick(tree.topology_root) + ";"
