"""Builder sanity tests."""

import numpy as np
import pytest

from hyperiax import symmetric_topology


def test_symmetric_height0():
    topo = symmetric_topology(depth=0, degree=2)
    assert topo.size == 1
    assert topo.depth == 0
    assert topo.is_root[0] and topo.is_leaf[0]


def test_symmetric_binary_height3_has_15_nodes():
    topo = symmetric_topology(depth=3, degree=2)
    assert topo.size == 15
    assert topo.depth == 3
    assert topo.equal_degree is True
    assert topo.max_degree == 2
    np.testing.assert_array_equal(
        topo.parents,
        [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
    )


def test_symmetric_ternary_height2_has_13_nodes():
    topo = symmetric_topology(depth=2, degree=3)
    # 1 + 3 + 9 = 13
    assert topo.size == 13
    assert topo.depth == 2
    assert topo.equal_degree is True
    assert topo.max_degree == 3


def test_symmetric_degree1_is_a_chain():
    topo = symmetric_topology(depth=3, degree=1)
    assert topo.size == 4
    assert topo.depth == 3
    np.testing.assert_array_equal(topo.parents, [0, 0, 1, 2])


def test_symmetric_rejects_negative_height():
    with pytest.raises(ValueError):
        symmetric_topology(depth=-1, degree=2)


def test_symmetric_rejects_zero_degree():
    with pytest.raises(ValueError):
        symmetric_topology(depth=2, degree=0)
