"""Unequal-degree up-sweep via ChildrenAxis. Covers T-7."""

import jax
import jax.numpy as jnp
import pytest

from hyperiax import (
    HyperiaxError,
    SchemaMismatch,
    Topology,
    Tree,
    symmetric_topology,
    up,
)

# Canonical unequal-degree tree used by the T-7 family:
#       0
#      / \
#     1   2
#    /|\  /\
#   3 4 5 6 7
# (node 1 has 3 children; node 2 has 2). depth=2.
PARENTS = [0, 0, 0, 1, 1, 1, 2, 2]


def _make_tree_with_leaf_values():
    topo = Topology.from_parents(PARENTS)
    tree = Tree.empty(topo, {"value": ()})
    leaf_vals = jnp.array([10, 20, 30, 100, 200], dtype=jnp.float32)
    tree = tree.at[topo.is_leaf].set(value=leaf_vals)
    return topo, tree


def test_unequal_topology_pbuckets_layout():
    """Sanity-check: pbuckets is local segment ids; pbuckets_ref is parent ids."""
    topo = Topology.from_parents(PARENTS)
    assert not topo.equal_degree
    # Level 2 children [3..7] reduce into parents [1, 2] via segments [0,0,0,1,1]
    assert list(topo.pbuckets[3:8]) == [0, 0, 0, 1, 1]
    assert list(topo.pbuckets_ref[2]) == [1, 2]


def test_unequal_up_sweep_sum_matches_segment_sum_by_hand():
    """T-7: children.value.sum(0) on the canonical ragged tree."""
    topo, tree = _make_tree_with_leaf_values()

    @up(reads_children=("value",), writes=("value",))
    def sum_up(node, children, params):
        return {"value": children.value.sum(0)}

    out = sum_up(tree)
    # Node 1 = 10+20+30 = 60; Node 2 = 100+200 = 300; Node 0 = 60+300 = 360.
    expected = jnp.array([360, 60, 300, 10, 20, 30, 100, 200], dtype=jnp.float32)
    assert jnp.allclose(out["value"], expected)


def test_unequal_up_sweep_mean():
    topo, tree = _make_tree_with_leaf_values()

    @up(reads_children=("value",), writes=("value",))
    def mean_up(node, children, params):
        return {"value": children.value.mean(0)}

    out = mean_up(tree)
    # Node 1 = mean(10,20,30) = 20; Node 2 = mean(100,200) = 150
    # Node 0 = mean(node1=20, node2=150) = 85
    assert jnp.allclose(out["value"][1], 20.0)
    assert jnp.allclose(out["value"][2], 150.0)
    assert jnp.allclose(out["value"][0], 85.0)


def test_unequal_up_sweep_max():
    topo, tree = _make_tree_with_leaf_values()

    @up(reads_children=("value",), writes=("value",))
    def max_up(node, children, params):
        return {"value": children.value.max(0)}

    out = max_up(tree)
    assert jnp.allclose(out["value"][1], 30.0)
    assert jnp.allclose(out["value"][2], 200.0)
    assert jnp.allclose(out["value"][0], 200.0)


def test_unequal_up_sweep_min():
    topo, tree = _make_tree_with_leaf_values()

    @up(reads_children=("value",), writes=("value",))
    def min_up(node, children, params):
        return {"value": children.value.min(0)}

    out = min_up(tree)
    assert jnp.allclose(out["value"][1], 10.0)
    assert jnp.allclose(out["value"][2], 100.0)
    assert jnp.allclose(out["value"][0], 10.0)


def test_unequal_up_sweep_prod():
    topo = Topology.from_parents(PARENTS)
    tree = Tree.empty(topo, {"value": ()})
    leaf_vals = jnp.array([2.0, 3.0, 5.0, 7.0, 11.0])
    tree = tree.at[topo.is_leaf].set(value=leaf_vals)

    @up(reads_children=("value",), writes=("value",))
    def prod_up(node, children, params):
        return {"value": children.value.prod(0)}

    out = prod_up(tree)
    # Node 1 = 2*3*5 = 30; Node 2 = 7*11 = 77; Node 0 = 30*77 = 2310
    assert jnp.allclose(out["value"][1], 30.0)
    assert jnp.allclose(out["value"][2], 77.0)
    assert jnp.allclose(out["value"][0], 2310.0)


def test_unequal_up_sweep_combines_node_and_children():
    """node.bias + children.value.sum(0) in unequal-degree mode."""
    topo = Topology.from_parents(PARENTS)
    tree = Tree.empty(topo, {"value": (), "bias": ()})
    tree = tree.at[topo.is_leaf].set(value=jnp.ones(5))
    tree = tree.set(bias=jnp.arange(8, dtype=jnp.float32))

    @up(reads=("bias",), reads_children=("value",), writes=("value",))
    def with_bias(node, children, params):
        return {"value": children.value.sum(0) + node.bias}

    out = with_bias(tree)
    # Node 1 (bias 1): sum(1,1,1) + 1 = 4
    # Node 2 (bias 2): sum(1,1)   + 2 = 4
    # Node 0 (bias 0): sum(4, 4)  + 0 = 8
    assert jnp.allclose(out["value"][1], 4.0)
    assert jnp.allclose(out["value"][2], 4.0)
    assert jnp.allclose(out["value"][0], 8.0)


def test_unequal_up_sweep_with_multidim_trailing():
    """The children axis works for arbitrary trailing shape."""
    topo = Topology.from_parents(PARENTS)
    tree = Tree.empty(topo, {"value": (2,)})
    # Each leaf gets a 2-vector; values chosen to be inspectable.
    leaf_vals = jnp.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]], dtype=jnp.float32)
    tree = tree.at[topo.is_leaf].set(value=leaf_vals)

    @up(reads_children=("value",), writes=("value",))
    def sum_up(node, children, params):
        return {"value": children.value.sum(0)}

    out = sum_up(tree)
    # Node 1 = [1+2+3, 10+20+30] = [6, 60]
    # Node 2 = [4+5, 40+50]      = [9, 90]
    # Node 0 = [6+9, 60+90]      = [15, 150]
    assert jnp.allclose(out["value"][1], jnp.array([6, 60]))
    assert jnp.allclose(out["value"][2], jnp.array([9, 90]))
    assert jnp.allclose(out["value"][0], jnp.array([15, 150]))


def test_unequal_up_sweep_with_mixed_depth_leaves():
    """A tree where some level-1 nodes are leaves (depth varies) AND the
    internal arities differ — the genuinely-ragged case."""
    # root 0 has children {1, 2}; node 1 has children {3, 4, 5} (leaves);
    # node 2 is itself a leaf at level 1. Internal arities differ: 2 vs 3.
    topo = Topology.from_parents([0, 0, 0, 1, 1, 1])
    assert not topo.equal_degree
    tree = Tree.empty(topo, {"value": ()})
    # is_leaf = [F, F, T, T, T, T] → leaves are at indices 2, 3, 4, 5.
    tree = tree.at[topo.is_leaf].set(value=jnp.array([7.0, 100.0, 200.0, 300.0]))

    @up(reads_children=("value",), writes=("value",))
    def sum_up(node, children, params):
        return {"value": children.value.sum(0)}

    out = sum_up(tree)
    # Node 1 reduces children {3,4,5} → 100 + 200 + 300 = 600
    # Node 2 is a leaf at level 1 → untouched, stays 7
    # Root reduces children {1, 2} → 600 + 7 = 607
    assert jnp.allclose(out["value"][1], 600.0)
    assert jnp.allclose(out["value"][2], 7.0)
    assert jnp.allclose(out["value"][0], 607.0)


# ── proxy guards ────────────────────────────────────────────────────
def test_children_axis_rejects_non_zero_axis():
    topo, tree = _make_tree_with_leaf_values()

    @up(reads_children=("value",), writes=("value",))
    def bad_axis(node, children, params):
        return {"value": children.value.sum(1)}  # not allowed on a proxy

    with pytest.raises(ValueError, match="axis=0"):
        bad_axis(tree)


def test_children_axis_rejects_array_coercion():
    topo, tree = _make_tree_with_leaf_values()

    @up(reads_children=("value",), writes=("value",))
    def coerce(node, children, params):
        # Trying to pull the proxy into a dense jnp array should hit __array__.
        return {"value": jnp.asarray(children.value)}

    with pytest.raises((HyperiaxError, TypeError, ValueError)):
        coerce(tree)


# ── JIT cache + outer jit ──────────────────────────────────────────
def test_unequal_up_sweep_jit_cache_hits_on_identical_topology():
    trace_count = 0

    @up(reads_children=("value",), writes=("value",))
    def sum_up(node, children, params):
        nonlocal trace_count
        trace_count += 1
        return {"value": children.value.sum(0)}

    topo1 = Topology.from_parents(PARENTS)
    tree1 = Tree.empty(topo1, {"value": ()}).at[topo1.is_leaf].set(value=jnp.ones(5))
    sum_up(tree1)["value"].block_until_ready()
    initial = trace_count
    assert initial > 0

    topo2 = Topology.from_parents(PARENTS)
    tree2 = Tree.empty(topo2, {"value": ()}).at[topo2.is_leaf].set(value=jnp.ones(5))
    for _ in range(5):
        sum_up(tree2)["value"].block_until_ready()
    assert trace_count == initial


def test_unequal_up_sweep_outer_jit_does_not_leak():
    topo, tree = _make_tree_with_leaf_values()

    @up(reads_children=("value",), writes=("value",))
    def sum_up(node, children, params):
        return {"value": children.value.sum(0)}

    @jax.jit
    def step(t):
        return sum_up(t)

    out_eager = sum_up(tree)
    out_jit = step(tree)
    assert jnp.allclose(out_eager["value"], out_jit["value"])


# ── equal/unequal API parity (same user code, both modes work) ─────
def test_same_user_code_runs_on_equal_and_unequal_topologies():
    """The same up function executes on a symmetric (equal-degree) tree
    AND on a ragged (unequal-degree) tree — proving the user-facing
    surface is the same. Numerically they differ because the trees do."""

    @up(reads_children=("value",), writes=("value",))
    def sum_up(node, children, params):
        return {"value": children.value.sum(0)}

    sym = symmetric_topology(depth=2, degree=2)  # equal-degree
    rag = Topology.from_parents(PARENTS)  # unequal-degree

    sym_tree = Tree.empty(sym, {"value": ()}).at[sym.is_leaf].set(value=jnp.ones(4))
    rag_tree = Tree.empty(rag, {"value": ()}).at[rag.is_leaf].set(value=jnp.ones(5))

    out_sym = sum_up(sym_tree)
    out_rag = sum_up(rag_tree)

    assert jnp.allclose(out_sym["value"][0], 4.0)  # 4 leaves, each 1
    assert jnp.allclose(out_rag["value"][0], 5.0)  # 5 leaves, each 1


# ── validation paths ───────────────────────────────────────────────
def test_unequal_up_sweep_raises_on_extra_writes():
    topo, tree = _make_tree_with_leaf_values()

    @up(reads_children=("value",), writes=("value",))
    def extra(node, children, params):
        return {"value": children.value.sum(0), "phantom": jnp.zeros(())}

    with pytest.raises(SchemaMismatch):
        extra(tree)
