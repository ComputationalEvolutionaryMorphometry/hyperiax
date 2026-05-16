"""Up-sweep dispatch on equal-degree trees. Covers T-4 (correctness) and T-5
(jit cache hit on structurally-identical trees).
"""

import jax
import jax.numpy as jnp
import pytest

import hyperiax as hx
from hyperiax import MissingField, SchemaMismatch, Tree, symmetric_topology, up


# ── T-4: numerical correctness ──────────────────────────────────────
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
    topo = symmetric_topology(depth=3, degree=2)  # leaves at [7..14]
    tree = Tree.empty(topo, {"value": (1,)})
    leaf_vals = jnp.arange(8, dtype=jnp.float32).reshape(8, 1)
    tree = tree.at[topo.is_leaf].set(value=leaf_vals)

    @up(reads_children=("value",), writes=("value",))
    def avg(node, children, params):
        return {"value": children.value.mean(0)}

    out = avg(tree)
    # Node 3 has leaves 7, 8 → mean 0.5; node 4 → 2.5; node 5 → 4.5; node 6 → 6.5
    assert jnp.allclose(out["value"][3], 0.5)
    assert jnp.allclose(out["value"][4], 2.5)
    assert jnp.allclose(out["value"][5], 4.5)
    assert jnp.allclose(out["value"][6], 6.5)
    # Node 1 = mean(0.5, 2.5) = 1.5; node 2 = mean(4.5, 6.5) = 5.5; root = 3.5
    assert jnp.allclose(out["value"][1], 1.5)
    assert jnp.allclose(out["value"][2], 5.5)
    assert jnp.allclose(out["value"][0], 3.5)


def test_up_sweep_passes_node_data_alongside_children():
    """The user's function gets both node and children data per parent."""
    topo = symmetric_topology(depth=2, degree=2)  # 7 nodes
    tree = Tree.empty(topo, {"value": (), "bias": ()})
    tree = tree.at[topo.is_leaf].set(value=jnp.ones(4))
    tree = tree.set(bias=jnp.arange(7, dtype=jnp.float32))  # 0..6

    @up(reads=("bias",), reads_children=("value",), writes=("value",))
    def sum_plus_bias(node, children, params):
        return {"value": children.value.sum(0) + node.bias}

    out = sum_plus_bias(tree)
    # leaves stay 1; level-1 parents: sum of 2 leaves + bias
    # node 1 bias=1, children sum = 2 → value=3
    # node 2 bias=2, children sum = 2 → value=4
    assert jnp.allclose(out["value"][1], 3.0)
    assert jnp.allclose(out["value"][2], 4.0)
    # root bias=0, children sum = 3+4 → value=7
    assert jnp.allclose(out["value"][0], 7.0)


def test_up_sweep_threads_params():
    """`params` reaches the user fn unchanged."""
    topo = symmetric_topology(depth=1, degree=2)
    tree = Tree.empty(topo, {"value": ()}).at[topo.is_leaf].set(value=jnp.ones(2))

    @up(reads_children=("value",), writes=("value",))
    def scaled_sum(node, children, params):
        return {"value": children.value.sum(0) * params["gain"]}

    out = scaled_sum(tree, params={"gain": jnp.array(3.0)})
    assert jnp.allclose(out["value"][0], 6.0)


# ── T-5: JIT cache hit on structurally-identical trees ─────────────
def test_up_sweep_jit_cache_hits_on_identical_topology():
    """Building 'the same' Tree twice must hit the JIT cache: the user
    function is traced only during the first compile."""
    trace_count = 0

    @up(reads_children=("value",), writes=("value",))
    def avg(node, children, params):
        nonlocal trace_count
        trace_count += 1
        return {"value": children.value.mean(0)}

    # Fresh topology + tree
    topo1 = symmetric_topology(depth=3, degree=2)
    tree1 = Tree.empty(topo1, {"value": (2,)}).at[topo1.is_leaf].set(value=jnp.ones((8, 2)))
    avg(tree1)["value"].block_until_ready()
    initial = trace_count
    assert initial > 0  # we did trace on first compile

    # Fresh tree with structurally-identical topology (different Python objects)
    topo2 = symmetric_topology(depth=3, degree=2)
    assert topo2 is not topo1
    assert topo2 == topo1

    tree2 = Tree.empty(topo2, {"value": (2,)}).at[topo2.is_leaf].set(value=jnp.ones((8, 2)))
    for _ in range(5):
        avg(tree2)["value"].block_until_ready()

    # No additional traces — the cached XLA binary handled all 5 calls.
    assert trace_count == initial, (
        f"Expected no extra traces (cache hit), but got {trace_count - initial} extras."
    )


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
    # New schema (trailing shape (3,)) → fresh trace
    assert trace_count > count_after_first


# ── validation paths ──────────────────────────────────────────────
def test_up_sweep_raises_on_missing_field():
    topo = symmetric_topology(depth=2, degree=2)
    tree = Tree.empty(topo, {"value": (2,)})

    @up(reads_children=("not_in_schema",), writes=("value",))
    def broken(node, children, params):
        return {"value": children.not_in_schema.mean(0)}

    with pytest.raises(MissingField):
        broken(tree)


def test_up_sweep_raises_on_extra_writes():
    """Returning a key not declared in writes is a schema violation."""
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
    """SweepFn (the internal primitive) refuses parent-reads on up sweeps."""
    with pytest.raises(ValueError):
        hx.SweepFn(
            direction="up", fn=lambda *a: {"value": None},
            reads=None, reads_children=None,
            reads_parent=("value",), writes=("value",),
        )
