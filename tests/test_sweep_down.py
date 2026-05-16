"""Down-sweep dispatch. Covers T-6."""

import jax
import jax.numpy as jnp
import pytest

import hyperiax as hx
from hyperiax import (
    MissingField,
    SchemaMismatch,
    Topology,
    Tree,
    down,
    symmetric_topology,
)


# ── T-6: propagation correctness ────────────────────────────────────
def test_down_sweep_propagates_root_value_via_delta():
    """``value <- parent.value + node.delta`` accumulates along path-to-root."""
    topo = symmetric_topology(depth=2, degree=2)  # 7 nodes
    tree = (
        Tree.empty(topo, {"value": (), "delta": ()})
        .set(delta=jnp.ones(7))
    )

    @down(reads=("delta",), reads_parent=("value",), writes=("value",))
    def propagate(node, parent, params):
        return {"value": parent.value + node.delta}

    out = propagate(tree)
    # root keeps its zero default; level 1 inherits +1; level 2 inherits +2.
    expected = jnp.array([0, 1, 1, 2, 2, 2, 2], dtype=jnp.float32)
    assert jnp.allclose(out["value"], expected)


def test_down_sweep_copies_root_to_all_descendants():
    topo = symmetric_topology(depth=3, degree=2)
    tree = Tree.empty(topo, {"value": (2,)}).at[topo.is_root].set(value=jnp.array([[3.0, 4.0]])
    )

    @down(reads_parent=("value",), writes=("value",))
    def copy_down(node, parent, params):
        return {"value": parent.value}

    out = copy_down(tree)
    assert jnp.all(out["value"] == jnp.array([3.0, 4.0]))


def test_down_sweep_threads_params():
    topo = symmetric_topology(depth=1, degree=3)
    tree = Tree.empty(topo, {"value": ()}).at[topo.is_root].set(value=jnp.array([2.0]))

    @down(reads_parent=("value",), writes=("value",))
    def scale(node, parent, params):
        return {"value": parent.value * params["gain"]}

    out = scale(tree, params={"gain": jnp.array(5.0)})
    # root=2 unchanged; 3 leaves each = 2*5 = 10
    assert jnp.allclose(out["value"][0], 2.0)
    assert jnp.allclose(out["value"][topo.is_leaf], 10.0)


def test_down_sweep_handles_unequal_degree_topology():
    """Down direction doesn't need the equal-degree fast path — it should
    work on any topology because every non-root node has exactly one
    parent. This is the asymmetric tree where node 1 has 3 children and
    node 2 has 0."""
    topo = Topology.from_parents([0, 0, 0, 1, 1, 1])
    assert not topo.equal_degree
    tree = Tree.empty(topo, {"value": ()}).at[topo.is_root].set(value=jnp.array([7.0]))

    @down(reads_parent=("value",), writes=("value",))
    def copy_down(node, parent, params):
        return {"value": parent.value}

    out = copy_down(tree)
    assert jnp.all(out["value"] == 7.0)


def test_down_sweep_root_remains_unchanged():
    """Root has no parent; the sweep starts at level 1, so root data is
    never touched by the user function."""
    topo = symmetric_topology(depth=2, degree=2)
    tree = Tree.empty(topo, {"value": ()}).at[topo.is_root].set(value=jnp.array([99.0])
    )

    @down(reads_parent=("value",), writes=("value",))
    def reset_to_zero(node, parent, params):
        return {"value": jnp.zeros(())}

    out = reset_to_zero(tree)
    assert jnp.allclose(out["value"][0], 99.0)
    assert jnp.allclose(out["value"][1:], 0.0)


# ── JIT cache + outer jit ──────────────────────────────────────────
def test_down_sweep_outer_jit_does_not_leak_tracer():
    topo = symmetric_topology(depth=2, degree=2)
    tree = Tree.empty(topo, {"value": (), "delta": ()}).set(delta=jnp.ones(7))

    @down(reads=("delta",), reads_parent=("value",), writes=("value",))
    def propagate(node, parent, params):
        return {"value": parent.value + node.delta}

    @jax.jit
    def step(t):
        return propagate(t)

    out_eager = propagate(tree)
    out_jit = step(tree)
    assert jnp.allclose(out_eager["value"], out_jit["value"])


def test_down_sweep_jit_cache_hits_on_identical_topology():
    trace_count = 0

    @down(reads_parent=("value",), writes=("value",))
    def copy_down(node, parent, params):
        nonlocal trace_count
        trace_count += 1
        return {"value": parent.value}

    topo1 = symmetric_topology(depth=3, degree=2)
    tree1 = Tree.empty(topo1, {"value": (2,)}).at[topo1.is_root].set(value=jnp.array([[1.0, 2.0]])
    )
    copy_down(tree1)["value"].block_until_ready()
    initial = trace_count
    assert initial > 0

    topo2 = symmetric_topology(depth=3, degree=2)
    tree2 = Tree.empty(topo2, {"value": (2,)}).at[topo2.is_root].set(value=jnp.array([[5.0, 6.0]])
    )
    for _ in range(5):
        copy_down(tree2)["value"].block_until_ready()
    assert trace_count == initial


# ── up + down round-trip ───────────────────────────────────────────
def test_up_then_down_broadcasts_leaf_mean_to_all_nodes():
    """A canonical pipeline: up-sweep computes the leaf mean at the root,
    down-sweep broadcasts it to every node. After both, every node holds
    the leaf mean."""
    topo = symmetric_topology(depth=3, degree=2)
    tree = Tree.empty(topo, {"value": (2,)})
    leaf_vals = jnp.arange(16, dtype=jnp.float32).reshape(8, 2)
    tree = tree.at[topo.is_leaf].set(value=leaf_vals)

    @hx.up(reads_children=("value",), writes=("value",))
    def avg(node, children, params):
        return {"value": children.value.mean(0)}

    @hx.down(reads_parent=("value",), writes=("value",))
    def copy_down(node, parent, params):
        return {"value": parent.value}

    after_up = avg(tree)
    after_down = copy_down(after_up)

    expected_root = leaf_vals.mean(0)
    assert jnp.allclose(after_up["value"][0], expected_root)
    # All nodes (including leaves, since copy_down overwrites them) hold the mean.
    assert jnp.all(after_down["value"] == expected_root)


# ── validation paths ───────────────────────────────────────────────
def test_down_sweep_raises_on_missing_field():
    topo = symmetric_topology(depth=2, degree=2)
    tree = Tree.empty(topo, {"value": ()})

    @down(reads_parent=("does_not_exist",), writes=("value",))
    def broken(node, parent, params):
        return {"value": parent.does_not_exist}

    with pytest.raises(MissingField):
        broken(tree)


def test_down_sweep_raises_on_extra_writes():
    topo = symmetric_topology(depth=2, degree=2)
    tree = Tree.empty(topo, {"value": ()})

    @down(reads_parent=("value",), writes=("value",))
    def extra(node, parent, params):
        return {"value": parent.value, "phantom": jnp.zeros(())}

    with pytest.raises(SchemaMismatch):
        extra(tree)


def test_down_decorator_requires_nonempty_writes():
    with pytest.raises(ValueError):
        @down(writes=())
        def empty(node, parent, params):
            return {}


def test_sweepfn_rejects_children_reads_on_down_direction():
    with pytest.raises(ValueError):
        hx.SweepFn(
            direction="down", fn=lambda *a: {"value": None},
            reads=None, reads_children=("value",),
            reads_parent=None, writes=("value",),
        )
