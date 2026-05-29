"""Up-sweep dispatch.

A single segment-based path handles any topology (equal- or unequal-degree).
Tests are organised by capability, not by topology — most tests run the same
body on both an equal-degree symmetric tree and a ragged tree and assert
each matches its hand-derived expectation.
"""

import jax
import jax.numpy as jnp
import pytest

import hyperiax as hx
from hyperiax import (
    HyperiaxError,
    MissingField,
    SchemaMismatch,
    Topology,
    Tree,
    symmetric_topology,
    up,
)

# Canonical ragged tree used throughout the unequal-degree cases:
#       0
#      / \
#     1   2
#    /|\  /\
#   3 4 5 6 7         (node 1 has 3 children; node 2 has 2)
RAGGED_PARENTS = [0, 0, 0, 1, 1, 1, 2, 2]


def _ragged_tree(values, schema=("value",)):
    topo = Topology.from_parents(RAGGED_PARENTS)
    fields = {k: () for k in schema}
    tree = Tree.empty(topo, fields).at[topo.is_leaf].set(value=jnp.asarray(values))
    return topo, tree


# ── reduction correctness on equal-degree trees ────────────────────
def test_up_sweep_root_is_mean_of_all_leaves():
    """Recursive child-averaging on a 2^h binary tree collapses to leaf mean."""
    topo = symmetric_topology(depth=3, degree=2)
    tree = Tree.empty(topo, {"value": (2,)})
    n_leaves = int(topo.is_leaf.sum())
    leaf_vals = jnp.arange(2 * n_leaves, dtype=jnp.float32).reshape(n_leaves, 2)
    tree = tree.at[topo.is_leaf].set(value=leaf_vals)

    @up(reads_children=("value",), writes=("value",))
    def avg(node, children, params):
        return {"value": children.value.mean(0)}

    out = avg(tree)
    assert jnp.allclose(out["value"][0], leaf_vals.mean(0))


def test_up_sweep_per_level_values_match_hand_computed():
    """Level-2 nodes (3,4,5,6) should each be the mean of their 2 leaf children."""
    topo = symmetric_topology(depth=3, degree=2)
    tree = Tree.empty(topo, {"value": (1,)})
    leaf_vals = jnp.arange(8, dtype=jnp.float32).reshape(8, 1)
    tree = tree.at[topo.is_leaf].set(value=leaf_vals)

    @up(reads_children=("value",), writes=("value",))
    def avg(node, children, params):
        return {"value": children.value.mean(0)}

    out = avg(tree)
    assert jnp.allclose(out["value"][3], 0.5)
    assert jnp.allclose(out["value"][4], 2.5)
    assert jnp.allclose(out["value"][5], 4.5)
    assert jnp.allclose(out["value"][6], 6.5)
    assert jnp.allclose(out["value"][1], 1.5)
    assert jnp.allclose(out["value"][2], 5.5)
    assert jnp.allclose(out["value"][0], 3.5)


# ── reductions on ragged trees (segment_*) ─────────────────────────
def test_sum_on_ragged_tree():
    topo, tree = _ragged_tree([10.0, 20.0, 30.0, 100.0, 200.0])

    @up(reads_children=("value",), writes=("value",))
    def sum_up(node, children, params):
        return {"value": children.value.sum(0)}

    out = sum_up(tree)
    # node 1 = 10+20+30 = 60; node 2 = 100+200 = 300; root = 60+300 = 360
    expected = jnp.array([360, 60, 300, 10, 20, 30, 100, 200], dtype=jnp.float32)
    assert jnp.allclose(out["value"], expected)


def test_mean_on_ragged_tree():
    topo, tree = _ragged_tree([10.0, 20.0, 30.0, 100.0, 200.0])

    @up(reads_children=("value",), writes=("value",))
    def mean_up(node, children, params):
        return {"value": children.value.mean(0)}

    out = mean_up(tree)
    # node 1 = mean(10,20,30) = 20; node 2 = mean(100,200) = 150; root = mean(20,150) = 85
    assert jnp.allclose(out["value"][1], 20.0)
    assert jnp.allclose(out["value"][2], 150.0)
    assert jnp.allclose(out["value"][0], 85.0)


def test_max_on_ragged_tree():
    topo, tree = _ragged_tree([10.0, 20.0, 30.0, 100.0, 200.0])

    @up(reads_children=("value",), writes=("value",))
    def max_up(node, children, params):
        return {"value": children.value.max(0)}

    out = max_up(tree)
    assert jnp.allclose(out["value"][1], 30.0)
    assert jnp.allclose(out["value"][2], 200.0)
    assert jnp.allclose(out["value"][0], 200.0)


def test_min_on_ragged_tree():
    topo, tree = _ragged_tree([10.0, 20.0, 30.0, 100.0, 200.0])

    @up(reads_children=("value",), writes=("value",))
    def min_up(node, children, params):
        return {"value": children.value.min(0)}

    out = min_up(tree)
    assert jnp.allclose(out["value"][1], 10.0)
    assert jnp.allclose(out["value"][2], 100.0)
    assert jnp.allclose(out["value"][0], 10.0)


def test_prod_on_ragged_tree():
    topo, tree = _ragged_tree([2.0, 3.0, 5.0, 7.0, 11.0])

    @up(reads_children=("value",), writes=("value",))
    def prod_up(node, children, params):
        return {"value": children.value.prod(0)}

    out = prod_up(tree)
    # node 1 = 2*3*5 = 30; node 2 = 7*11 = 77; root = 30*77 = 2310
    assert jnp.allclose(out["value"][1], 30.0)
    assert jnp.allclose(out["value"][2], 77.0)
    assert jnp.allclose(out["value"][0], 2310.0)


def test_ragged_tree_with_mixed_depth_leaves():
    """A tree where some level-1 nodes are leaves (depth varies) AND the
    internal arities differ — the genuinely-ragged case."""
    # root 0 → {1, 2}; node 1 → {3, 4, 5} (leaves); node 2 itself a leaf at level 1.
    topo = Topology.from_parents([0, 0, 0, 1, 1, 1])
    assert not topo.equal_degree
    tree = Tree.empty(topo, {"value": ()})
    # is_leaf = [F, F, T, T, T, T] → leaves are at indices 2, 3, 4, 5.
    tree = tree.at[topo.is_leaf].set(value=jnp.array([7.0, 100.0, 200.0, 300.0]))

    @up(reads_children=("value",), writes=("value",))
    def sum_up(node, children, params):
        return {"value": children.value.sum(0)}

    out = sum_up(tree)
    # node 1 reduces children {3,4,5} → 600;  node 2 is a leaf at level 1 → 7;
    # root reduces children {1, 2} → 607.
    assert jnp.allclose(out["value"][1], 600.0)
    assert jnp.allclose(out["value"][2], 7.0)
    assert jnp.allclose(out["value"][0], 607.0)


# ── node + children combination ────────────────────────────────────
def test_up_sweep_passes_node_data_alongside_children():
    """The user fn gets both node and children data."""
    topo = symmetric_topology(depth=2, degree=2)
    tree = Tree.empty(topo, {"value": (), "bias": ()})
    tree = tree.at[topo.is_leaf].set(value=jnp.ones(4))
    tree = tree.set(bias=jnp.arange(7, dtype=jnp.float32))

    @up(reads=("bias",), reads_children=("value",), writes=("value",))
    def sum_plus_bias(node, children, params):
        return {"value": children.value.sum(0) + node.bias}

    out = sum_plus_bias(tree)
    assert jnp.allclose(out["value"][1], 3.0)  # bias=1, 2 leaves → 1+2
    assert jnp.allclose(out["value"][2], 4.0)
    assert jnp.allclose(out["value"][0], 7.0)


def test_node_plus_children_on_ragged_tree():
    topo = Topology.from_parents(RAGGED_PARENTS)
    tree = Tree.empty(topo, {"value": (), "bias": ()})
    tree = tree.at[topo.is_leaf].set(value=jnp.ones(5))
    tree = tree.set(bias=jnp.arange(8, dtype=jnp.float32))

    @up(reads=("bias",), reads_children=("value",), writes=("value",))
    def with_bias(node, children, params):
        return {"value": children.value.sum(0) + node.bias}

    out = with_bias(tree)
    # node 1 (bias=1): sum(1,1,1) + 1 = 4 ;  node 2 (bias=2): sum(1,1) + 2 = 4
    # root  (bias=0): sum(4, 4)   + 0 = 8
    assert jnp.allclose(out["value"][1], 4.0)
    assert jnp.allclose(out["value"][2], 4.0)
    assert jnp.allclose(out["value"][0], 8.0)


def test_ragged_tree_with_multidim_trailing():
    topo = Topology.from_parents(RAGGED_PARENTS)
    tree = Tree.empty(topo, {"value": (2,)})
    leaf_vals = jnp.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]], dtype=jnp.float32)
    tree = tree.at[topo.is_leaf].set(value=leaf_vals)

    @up(reads_children=("value",), writes=("value",))
    def sum_up(node, children, params):
        return {"value": children.value.sum(0)}

    out = sum_up(tree)
    assert jnp.allclose(out["value"][1], jnp.array([6, 60]))
    assert jnp.allclose(out["value"][2], jnp.array([9, 90]))
    assert jnp.allclose(out["value"][0], jnp.array([15, 150]))


# ── params threading ───────────────────────────────────────────────
def test_up_sweep_threads_params():
    topo = symmetric_topology(depth=1, degree=2)
    tree = Tree.empty(topo, {"value": ()}).at[topo.is_leaf].set(value=jnp.ones(2))

    @up(reads_children=("value",), writes=("value",))
    def scaled_sum(node, children, params):
        return {"value": children.value.sum(0) * params["gain"]}

    out = scaled_sum(tree, params={"gain": jnp.array(3.0)})
    assert jnp.allclose(out["value"][0], 6.0)


# ── children.map (per-child transform) ─────────────────────────────
def _sq1(c):
    """Non-linear per-child map: v -> v**2 + 1 (cannot be expressed as a plain
    segment reduction of the raw field)."""
    return {"value": c.value**2 + 1.0}


def test_map_then_sum_on_ragged_tree():
    topo, tree = _ragged_tree([2.0, 3.0, 4.0, 5.0, 6.0])

    @up(reads_children=("value",), writes=("value",))
    def sweep(node, children, params):
        return {"value": children.map(_sq1).value.sum(0)}

    out = sweep(tree)
    f = lambda v: v**2 + 1.0
    # node 1 = f(2)+f(3)+f(4); node 2 = f(5)+f(6); root = f(node1) + f(node2)
    assert jnp.allclose(out["value"][1], f(2.0) + f(3.0) + f(4.0))
    assert jnp.allclose(out["value"][2], f(5.0) + f(6.0))
    assert jnp.allclose(out["value"][0], f(32.0) + f(63.0))


def test_same_map_sweep_runs_on_equal_and_unequal():
    """One user fn, both topologies — exactly the same body."""

    @up(reads_children=("value",), writes=("value",))
    def sweep(node, children, params):
        return {"value": children.map(_sq1).value.sum(0)}

    f = lambda v: v**2 + 1.0

    sym = symmetric_topology(depth=2, degree=2)  # root>1,2 ; 1>3,4 ; 2>5,6
    sym_tree = Tree.empty(sym, {"value": ()}).at[sym.is_leaf].set(
        value=jnp.array([2.0, 3.0, 4.0, 5.0])
    )
    out_sym = sweep(sym_tree)
    assert jnp.allclose(out_sym["value"][1], f(2.0) + f(3.0))
    assert jnp.allclose(out_sym["value"][2], f(4.0) + f(5.0))
    assert jnp.allclose(out_sym["value"][0], f(15.0) + f(43.0))

    _, rag_tree = _ragged_tree([2.0, 3.0, 4.0, 5.0, 6.0])
    out_rag = sweep(rag_tree)
    assert jnp.allclose(out_rag["value"][0], f(32.0) + f(63.0))


def test_map_then_mean_and_max_stay_exact_on_ragged_tree():
    """After a per-child transform, segment-aware ``.mean``/``.max`` are
    still exact (no padding-identity assumption)."""
    topo, tree = _ragged_tree([2.0, 3.0, 4.0, 5.0, 6.0])

    @up(reads_children=("value",), writes=("value",))
    def mean_sweep(node, children, params):
        return {"value": children.map(lambda c: {"value": c.value * 10.0}).value.mean(0)}

    @up(reads_children=("value",), writes=("value",))
    def max_sweep(node, children, params):
        return {"value": children.map(lambda c: {"value": c.value * 10.0}).value.max(0)}

    out_mean = mean_sweep(tree)
    assert jnp.allclose(out_mean["value"][1], 30.0)  # mean([2,3,4]*10)
    assert jnp.allclose(out_mean["value"][2], 55.0)  # mean([5,6]*10)

    out_max = max_sweep(tree)
    assert jnp.allclose(out_max["value"][1], 40.0)
    assert jnp.allclose(out_max["value"][2], 60.0)


# ── writes_children (per-edge scatter) ─────────────────────────────
def test_writes_children_scatters_per_child_equal_degree():
    topo = symmetric_topology(depth=2, degree=2)  # nodes 0..6, leaves 3..6
    tree = Tree.empty(topo, {"value": (), "doubled": ()}).at[topo.is_leaf].set(
        value=jnp.array([2.0, 3.0, 4.0, 5.0])
    )

    @up(reads_children=("value",), writes=("value",), writes_children=("doubled",))
    def sweep(node, children, params):
        msgs = children.map(lambda c: {"value": c.value, "doubled": 2.0 * c.value})
        return {"value": msgs.value.sum(0), "doubled": msgs.doubled}

    out = sweep(tree)
    # node 1 = 5; node 2 = 9; root = 14.
    assert jnp.allclose(out["value"][0], 14.0)
    assert jnp.allclose(out["value"][1], 5.0)
    assert jnp.allclose(out["value"][2], 9.0)
    # leaves get 2*value; nodes 1,2 get 2*(fused value); root never written.
    assert jnp.allclose(out["doubled"][3:7], jnp.array([4.0, 6.0, 8.0, 10.0]))
    assert jnp.allclose(out["doubled"][1], 10.0)
    assert jnp.allclose(out["doubled"][2], 18.0)
    assert jnp.allclose(out["doubled"][0], 0.0)


def test_writes_children_on_ragged_tree():
    topo, tree = _ragged_tree([2.0, 3.0, 4.0, 5.0, 6.0], schema=("value", "doubled"))

    @up(reads_children=("value",), writes=("value",), writes_children=("doubled",))
    def sweep(node, children, params):
        msgs = children.map(lambda c: {"value": c.value, "doubled": 2.0 * c.value})
        return {"value": msgs.value.sum(0), "doubled": msgs.doubled}

    out = sweep(tree)
    # node 1 = 9; node 2 = 11; root = 20.
    assert jnp.allclose(out["value"][1], 9.0)
    assert jnp.allclose(out["value"][2], 11.0)
    assert jnp.allclose(out["value"][0], 20.0)
    # per-child scatter
    expected = jnp.array([0.0, 2 * 9.0, 2 * 11.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    assert jnp.allclose(out["doubled"], expected)


def test_writes_children_on_unbalanced_ragged_tree():
    """Single-child internal node: writes_children scatter still lands correctly."""
    topo = Topology.from_parents([0, 0, 0, 0, 1])  # root has 3 children, node 1 has 1
    assert not topo.equal_degree
    tree = Tree.empty(topo, {"value": (), "doubled": ()}).at[topo.is_leaf].set(
        value=jnp.array([2.0, 3.0, 4.0])  # leaves 2, 3, 4
    )

    @up(reads_children=("value",), writes=("value",), writes_children=("doubled",))
    def sweep(node, children, params):
        msgs = children.map(lambda c: {"value": c.value, "doubled": 2.0 * c.value})
        return {"value": msgs.value.sum(0), "doubled": msgs.doubled}

    out = sweep(tree)
    # node 1's only child is leaf 4 → node1.value = 4;  doubled[4] = 8
    # root's children are 1 (=4), 2 (=2), 3 (=3) → root.value = 9
    assert jnp.allclose(out["value"][1], 4.0)
    assert jnp.allclose(out["value"][0], 9.0)
    assert jnp.allclose(out["doubled"][2], 4.0)
    assert jnp.allclose(out["doubled"][3], 6.0)
    assert jnp.allclose(out["doubled"][4], 8.0)
    assert jnp.allclose(out["doubled"][1], 8.0)


# ── ChildrenAxis proxy guards ──────────────────────────────────────
def test_children_axis_rejects_non_zero_axis():
    topo, tree = _ragged_tree([10.0, 20.0, 30.0, 100.0, 200.0])

    @up(reads_children=("value",), writes=("value",))
    def bad_axis(node, children, params):
        return {"value": children.value.sum(1)}  # only axis=0 is allowed

    with pytest.raises(ValueError, match="axis=0"):
        bad_axis(tree)


def test_children_axis_rejects_array_coercion():
    topo, tree = _ragged_tree([10.0, 20.0, 30.0, 100.0, 200.0])

    @up(reads_children=("value",), writes=("value",))
    def coerce(node, children, params):
        # Pulling the proxy into a dense jnp array should hit __array__ and fail.
        return {"value": jnp.asarray(children.value)}

    with pytest.raises((HyperiaxError, TypeError, ValueError)):
        coerce(tree)


# ── JIT cache + outer JIT ──────────────────────────────────────────
def test_up_sweep_jit_cache_hits_on_identical_equal_degree_topology():
    trace_count = 0

    @up(reads_children=("value",), writes=("value",))
    def avg(node, children, params):
        nonlocal trace_count
        trace_count += 1
        return {"value": children.value.mean(0)}

    topo1 = symmetric_topology(depth=3, degree=2)
    tree1 = Tree.empty(topo1, {"value": (2,)}).at[topo1.is_leaf].set(value=jnp.ones((8, 2)))
    avg(tree1)["value"].block_until_ready()
    initial = trace_count
    assert initial > 0

    topo2 = symmetric_topology(depth=3, degree=2)
    assert topo2 is not topo1 and topo2 == topo1
    tree2 = Tree.empty(topo2, {"value": (2,)}).at[topo2.is_leaf].set(value=jnp.ones((8, 2)))
    for _ in range(5):
        avg(tree2)["value"].block_until_ready()
    assert trace_count == initial


def test_up_sweep_jit_cache_hits_on_identical_ragged_topology():
    trace_count = 0

    @up(reads_children=("value",), writes=("value",))
    def sum_up(node, children, params):
        nonlocal trace_count
        trace_count += 1
        return {"value": children.value.sum(0)}

    topo1 = Topology.from_parents(RAGGED_PARENTS)
    tree1 = Tree.empty(topo1, {"value": ()}).at[topo1.is_leaf].set(value=jnp.ones(5))
    sum_up(tree1)["value"].block_until_ready()
    initial = trace_count
    assert initial > 0

    topo2 = Topology.from_parents(RAGGED_PARENTS)
    tree2 = Tree.empty(topo2, {"value": ()}).at[topo2.is_leaf].set(value=jnp.ones(5))
    for _ in range(5):
        sum_up(tree2)["value"].block_until_ready()
    assert trace_count == initial


def test_up_sweep_does_recompile_for_different_schema():
    """Different schema (different pytree structure) → separate compile."""
    trace_count = 0

    @up(reads_children=("value",), writes=("value",))
    def avg(node, children, params):
        nonlocal trace_count
        trace_count += 1
        return {"value": children.value.mean(0)}

    topo = symmetric_topology(depth=2, degree=2)
    t_2d = Tree.empty(topo, {"value": (2,)}).at[topo.is_leaf].set(value=jnp.ones((4, 2)))
    t_3d = Tree.empty(topo, {"value": (3,)}).at[topo.is_leaf].set(value=jnp.ones((4, 3)))
    avg(t_2d)["value"].block_until_ready()
    count_after_first = trace_count
    avg(t_3d)["value"].block_until_ready()
    assert trace_count > count_after_first


def test_up_sweep_outer_jit_does_not_leak_on_ragged_tree():
    topo, tree = _ragged_tree([10.0, 20.0, 30.0, 100.0, 200.0], schema=("value", "doubled"))

    @up(reads_children=("value",), writes=("value",), writes_children=("doubled",))
    def sweep(node, children, params):
        msgs = children.map(lambda c: {"value": c.value**2, "doubled": 2.0 * c.value})
        return {"value": msgs.value.sum(0), "doubled": msgs.doubled}

    @jax.jit
    def step(t):
        return sweep(t)

    out_eager = sweep(tree)
    out_jit = step(tree)
    assert jnp.allclose(out_eager["value"], out_jit["value"])
    assert jnp.allclose(out_eager["doubled"], out_jit["doubled"])


# ── validation paths ───────────────────────────────────────────────
def test_up_sweep_raises_on_missing_field():
    topo = symmetric_topology(depth=2, degree=2)
    tree = Tree.empty(topo, {"value": (2,)})

    @up(reads_children=("not_in_schema",), writes=("value",))
    def broken(node, children, params):
        return {"value": children.not_in_schema.mean(0)}

    with pytest.raises(MissingField):
        broken(tree)


def test_up_sweep_raises_on_extra_writes():
    topo = symmetric_topology(depth=2, degree=2)
    tree = Tree.empty(topo, {"value": (2,)}).at[topo.is_leaf].set(value=jnp.ones((4, 2)))

    @up(reads_children=("value",), writes=("value",))
    def extra_key(node, children, params):
        return {"value": children.value.mean(0), "phantom": jnp.zeros((2,))}

    with pytest.raises(SchemaMismatch):
        extra_key(tree)


def test_up_decorator_requires_nonempty_writes():
    with pytest.raises(ValueError):

        @up(writes=())
        def empty(node, children, params):
            return {}


def test_sweepfn_rejects_parent_reads_on_up_direction():
    with pytest.raises(ValueError):
        hx.SweepFn(
            direction="up",
            fn=lambda *a: {"value": None},
            reads=None,
            reads_children=None,
            reads_parent=("value",),
            writes=("value",),
        )


def test_writes_children_rejected_on_down_direction():
    with pytest.raises(ValueError, match="writes_children"):
        hx.SweepFn(
            direction="down",
            fn=lambda *a: {"value": None},
            reads=None,
            reads_children=None,
            reads_parent=None,
            writes=("value",),
            writes_children=("edge_id",),
        )


def test_writes_children_overlap_with_writes_rejected():
    with pytest.raises(ValueError, match="share fields"):
        hx.SweepFn(
            direction="up",
            fn=lambda *a: {"value": None},
            reads=None,
            reads_children=None,
            reads_parent=None,
            writes=("value",),
            writes_children=("value",),
        )
