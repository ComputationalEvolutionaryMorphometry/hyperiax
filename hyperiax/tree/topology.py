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
def read_topology(newick_str: str,return_topology=False) -> HypTree:
    """ 
    Generate a tree from a Newick string recursively.

    :param newick_str: newick string representation
    :return: The constructed tree
    """

    def parse_newick(newick_str):
        edge_lengths_collected = []
        k = -1
        root = current_node = TopologyNode(name=None, parent=None, children=[])  # Start with a root node
        for i, char in enumerate(newick_str):
            if i < k:
                continue
            elif char == '(':
                # Create a new node and make it a child of the current node
                new_node = TopologyNode(name=None, parent=current_node, children=[])
                current_node.children += [new_node]
                current_node = new_node  # Move down to the new node

            elif char == ',':
                # Go up to the parent, and then create a sibling node
                current_node = current_node.parent
                new_node = TopologyNode(name=None, parent=current_node, children=[])
                current_node.children += [new_node]
                current_node = new_node  # Move to the new sibling node

            elif char == ')':
                # End of a subtree, move up to the parent node
                current_node = current_node.parent
            elif char == ':':
                # Branch length follows
                start = i + 1
                while i + 1 < len(newick_str) and newick_str[i + 1] not in [',', ')', ';', ' ']:
                    i += 1
                edge_lengths_collected.append(float(newick_str[start:i + 1]))
                k = i+1

            elif char not in [';', ' ',":"] and newick_str[i-1] not in [':']:
                # Reading a node name, collect all characters till we hit a control character
                start = i
                while i + 1 < len(newick_str) and newick_str[i + 1] not in ['(', ')', ',', ';', ':', ' ']:
                    i += 1
                current_node.name = newick_str[start:i + 1]
                k = i+1
                
        return root, edge_lengths_collected

    root, edge_lengths_collected = parse_newick(newick_str)


    if return_topology:
        return root

    else:
        tree = HypTree(root)

        if len(edge_lengths_collected) > 0:
            tree.add_property('edge_length', (1,), dtype=float)

            # Assign edge lengths to nodes in post transversal order
            for i, node in enumerate(list(tree.iter_topology_post())[:-1]):
                tree.data['edge_length'] = tree.data['edge_length'].at[node.id].set(edge_lengths_collected[i])
     
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
