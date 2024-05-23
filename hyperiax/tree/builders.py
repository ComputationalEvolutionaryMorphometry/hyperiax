from . import HypTree, TreeNode
from .childrenlist import ChildList


def symmetric_tree(h: int, degree: int, new_node: any=TreeNode, fake_root: any=None) -> HypTree:
    """ 
    Generate a tree of given height and degree.
    A tree of height zero contains just the root;
    a tree of height one contains the root and one level of leaves below it, and so forth.

    :param h: The height of the tree
    :param degree: The degree of each node in the tree
    :param new_node: The node used to construct the tree, defaults to TreeNode
    :param fake_root: The fake root node, defaults to None
    :raises ValueError: If height is negative
    :return: The constructed tree
    """
    if h < 0:
        raise ValueError(f'Height shall be nonnegative integer, received {h=}.')

    def _builder(h: int, degree: int, parent):
        node = new_node(); node.parent = parent; node.children = ChildList()
        if h > 1:
            node.children = ChildList([_builder(h - 1, degree, node) for _ in range(degree)])
        return node

    if fake_root is None:
        return HypTree(root=_builder(h + 1, degree, None))
    else:
        fake_root.children = ChildList([_builder(h + 1, degree, fake_root)])
        return HypTree(root=fake_root)

def asymmetric_tree(h: int) -> HypTree:
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
        return HypTree(ChildList([TreeNode()]))

    # Fake root 
    root = TreeNode(); root.parent = None; root.children = ChildList()
    root.children = ChildList([TreeNode(children=ChildList(),parent=root), TreeNode(children=ChildList(),parent=root)])
   
    node = root.children[0]

    for _ in range(h - 1):
       # node.children = ChildList()
        node.children=  ChildList([TreeNode(children=ChildList(),parent=node), TreeNode(children=ChildList(),parent=node)])

        node = node.children[0]

    tree = HypTree(root)
    return tree 

### Tree generation and initialization
def tree_from_newick(newick_str: str) -> HypTree:
    """ 
    Generate a tree from a Newick string.

    :param newick_str: newick string representation
    :return: The constructed tree
    """
    def parse_newick(newick_str):
        k = -1
        root = current_node = TreeNode(children=[])  # Start with a root node
        for i, char in enumerate(newick_str):
            if i < k:
                #For some reason, the i and char will not be overwritten and contuine from there in the loop 
                # so we make this to skip where i<k, where k is the position of the last letter or number we use  
                continue
            elif char == '(':
                # Create a new node and make it a child of the current node
                new_node = TreeNode(parent=current_node,children =[])
                current_node.children += [new_node]
                current_node = new_node  # Move down to the new node

            elif char == ',':
                # Go up to the parent, and then create a sibling node
                current_node = current_node.parent
                new_node = TreeNode(parent=current_node,children =[])
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
                current_node.data["edge_length"] = float(newick_str[start:i + 1])
                k = i+1

            elif char not in [';', ' ',":"] and newick_str[i-1] not in [':']: # This does not work
                # Reading a node name, collect all characters till we hit a control character
                start = i

                while i + 1 < len(newick_str) and newick_str[i + 1] not in ['(', ')', ',', ';', ':', ' ']:
                    i += 1

                current_node.name = newick_str[start:i + 1]

                # For some reason this will not overwrite in the loop.... 
                #i  = i;char = newick_str[i+1]  # Adjust because the outer loop will increment `i`
                k = i+1
                
        return HypTree(root)
    return parse_newick(newick_str)


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

        node = TreeNode(name=name if name else None,             # create a "ghost" subtree root node without data
                       data={"edge_length": float(length)} if length else {},
                       parent=parent,
                       children=[])
        
        if char == "(": # start a subtree

            while char in "(,": # add all children within a parenthesis to the current node
                child, char = recursive_parse_newick(parent=node)
                node.children.append(child)

            name, length, delim, char = next(iter_tokens).groups(0)

            node.name = name                        # assign data to the "ghost" subtree root node
            node.data = {"edge_length": float(length)} if length else {}
            
        return node, delim
    
    return HypTree(recursive_parse_newick()[0])

