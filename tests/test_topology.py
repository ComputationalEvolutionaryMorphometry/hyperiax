"""Topology construction + pytree behavior. Covers T-1."""

import jax
import numpy as np
import pytest

from hyperiax import Topology


def test_from_parents_basic_structure():
    # 3-level binary tree:
    #          0
    #         / \
    #        1   2
    #       / \ / \
    #      3  4 5  6
    topo = Topology.from_parents([0, 0, 0, 1, 1, 2, 2])
    assert topo.size == 7
    assert topo.depth == 2
    np.testing.assert_array_equal(topo.level_starts, [0, 1, 3, 7])
    np.testing.assert_array_equal(topo.child_counts, [2, 2, 2, 0, 0, 0, 0])
    np.testing.assert_array_equal(topo.is_leaf, [False, False, False, True, True, True, True])
    np.testing.assert_array_equal(topo.is_root, [True, False, False, False, False, False, False])
    np.testing.assert_array_equal(topo.is_inner, [False, True, True, False, False, False, False])
    np.testing.assert_array_equal(topo.node_depths, [0, 1, 1, 2, 2, 2, 2])


def test_from_parents_detects_equal_degree():
    topo = Topology.from_parents([0, 0, 0, 1, 1, 2, 2])
    assert topo.equal_degree is True
    assert topo.max_degree == 2


def test_from_parents_unequal_degree():
    # Node 0 has 2 children; node 1 has 3 children; node 2 is a leaf with 0.
    topo = Topology.from_parents([0, 0, 0, 1, 1, 1])
    assert topo.equal_degree is False
    assert topo.max_degree == 3


def test_from_parents_segment_layout():
    topo = Topology.from_parents([0, 0, 0, 1, 1, 2, 2])
    # Level 2 (leaves) reduces into parents [1, 2]; their local seg ids are [0,0,1,1].
    np.testing.assert_array_equal(topo.pbuckets[3:7], [0, 0, 1, 1])
    np.testing.assert_array_equal(topo.pbuckets_ref[2], [1, 2])
    # Level 1 reduces into the root [0]; seg ids [0, 0].
    np.testing.assert_array_equal(topo.pbuckets[1:3], [0, 0])
    np.testing.assert_array_equal(topo.pbuckets_ref[1], [0])


def test_from_parents_segment_layout_on_ragged_tree():
    # node 1 has 3 children {3,4,5}; node 2 has 2 children {6,7}.
    topo = Topology.from_parents([0, 0, 0, 1, 1, 1, 2, 2])
    assert not topo.equal_degree
    # Level 2 children [3..7] reduce into parents [1, 2] with seg ids [0,0,0,1,1].
    np.testing.assert_array_equal(topo.pbuckets[3:8], [0, 0, 0, 1, 1])
    np.testing.assert_array_equal(topo.pbuckets_ref[2], [1, 2])


def test_from_parents_rejects_non_bfs_layout():
    # parents[2] = 3 points forward → not BFS.
    with pytest.raises(ValueError):
        Topology.from_parents([0, 0, 3, 0])


def test_from_parents_rejects_non_zero_root():
    with pytest.raises(ValueError):
        Topology.from_parents([1, 0])


def test_topology_single_node():
    topo = Topology.from_parents([0])
    assert topo.size == 1
    assert topo.depth == 0
    assert topo.is_root[0] and topo.is_leaf[0]
    assert topo.equal_degree is False


def test_topology_is_hashable_and_structurally_equal():
    a = Topology.from_parents([0, 0, 0, 1, 1, 2, 2])
    b = Topology.from_parents([0, 0, 0, 1, 1, 2, 2])
    assert hash(a) == hash(b)
    assert a == b
    c = Topology.from_parents([0, 0, 0])  # different shape
    assert a != c
    assert hash(a) != hash(c)


def test_topology_is_pytree_with_no_dynamic_leaves():
    topo = Topology.from_parents([0, 0, 0, 1, 1, 2, 2])
    leaves, treedef = jax.tree_util.tree_flatten(topo)
    assert leaves == []
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt == topo


def test_topology_all_derived_arrays_are_numpy_not_jax():
    """Every derived field must be ``np.ndarray`` so the Topology is hashable
    and rides through ``jax.jit`` as static aux_data."""
    topo = Topology.from_parents([0, 0, 0, 1, 1, 2, 2])
    for field_name in [
        "parents",
        "level_starts",
        "node_depths",
        "child_counts",
        "is_root",
        "is_leaf",
        "is_inner",
        "pbuckets",
    ]:
        val = getattr(topo, field_name)
        assert isinstance(val, np.ndarray), f"{field_name} should be np.ndarray, got {type(val)}"
