from hyperiax.tree.childrenlist import ChildList
from hyperiax.tree.tree import TreeNode
import pytest
from copy import deepcopy

def test_childlist():
    p1 = TreeNode()
    c1 = TreeNode()
    c2 = TreeNode()
    c3 = TreeNode()
    clist = ChildList([c1,c2])

    p1.children = clist

    with pytest.raises(ValueError):
        p1.children.append(TreeNode())

    with pytest.raises(ValueError):
        p1.children[0] = TreeNode()

    with pytest.raises(ValueError):
        p1.children += [TreeNode()]

    p1.add_child(c3)

    assert len(clist) == 3

    del p1.children[2]

    assert len(clist) == 2

    cp = deepcopy(clist)

    del cp[0]

    assert len(clist) != len(cp)