"""Regression tests for the v3 functional-purity story.

These are the tests that justify the whole refactor. The legacy v1/v2
``OrderedExecutor`` mutated ``tree.data = {...}`` in place; under an
outer ``@jax.jit`` (e.g. a Flax + optax ``train_step``) this leaked
tracers into Python state. The immutable Tree pytree fixes it.

Coverage:
- T-10: outer-jit composition (repeated calls, multi-sweep chains)
- T-11: lax.scan composition (body traced once regardless of length)
- jax.grad through up + down sweeps (no in-place state to break autodiff)
- Tree.update + sweep + jit (schema growth interacts cleanly with jit cache)
"""

import jax
import jax.numpy as jnp
import pytest

import hyperiax as hx
from hyperiax import Topology, Tree, down, symmetric_topology, up


# ── T-10: outer-jit composition ─────────────────────────────────────
def test_outer_jit_repeated_calls_no_leaked_tracer():
    """The historic failure mode: a ``step`` function wrapping a sweep,
    called repeatedly under ``@jax.jit``. The old in-place
    ``tree.data = {...}`` would leak tracers between calls. The immutable
    pytree Tree fixes it."""
    topo = symmetric_topology(depth=3, degree=2)

    @up(reads_children=("value",), writes=("value",))
    def avg(node, children, params):
        return {"value": children.value.mean(0)}

    @jax.jit
    def step(tree):
        return avg(tree)

    tree = Tree.empty(topo, {"value": (2,)}).at[topo.is_leaf].set(value=jnp.arange(16, dtype=jnp.float32).reshape(8, 2)
    )

    # 20 successive calls; old hyperiax would raise LeakedTracer within a few.
    for _ in range(20):
        tree = step(tree)
        assert jnp.all(jnp.isfinite(tree["value"]))


def test_outer_jit_composes_multi_sweep_pipeline():
    """A train-step-style function chaining up + down + up inside one jit.
    Each sweep returns a new Tree pytree; the chain composes without
    intermediate Python-state mutation."""
    topo = symmetric_topology(depth=3, degree=2)

    @up(reads_children=("value",), writes=("value",))
    def avg(node, children, params):
        return {"value": children.value.mean(0)}

    @down(reads_parent=("value",), writes=("value",))
    def copy_down(node, parent, params):
        return {"value": parent.value}

    @jax.jit
    def step(tree):
        t = avg(tree)        # root <- mean of leaves
        t = copy_down(t)     # broadcast back down
        t = avg(t)           # converged
        return t

    tree = Tree.empty(topo, {"value": (2,)}).at[topo.is_leaf].set(value=jnp.arange(16, dtype=jnp.float32).reshape(8, 2)
    )
    leaf_mean = jnp.arange(16, dtype=jnp.float32).reshape(8, 2).mean(0)

    out = step(tree)
    # After up→down, every node holds the leaf mean. The second up averages
    # k identical values per parent — still the leaf mean.
    assert jnp.allclose(out["value"][0], leaf_mean)
    assert jnp.all(jnp.abs(out["value"] - leaf_mean) < 1e-5)


def test_outer_jit_yields_same_result_as_eager():
    """`@jax.jit` is purely an optimization; numerical results must match
    the eager path bit-for-bit."""
    topo = symmetric_topology(depth=2, degree=2)
    tree = Tree.empty(topo, {"value": (), "delta": ()}).set(delta=jnp.ones(7))

    @down(reads=("delta",), reads_parent=("value",), writes=("value",))
    def propagate(node, parent, params):
        return {"value": parent.value + node.delta}

    eager = propagate(tree)
    jit_fn = jax.jit(lambda t: propagate(t))
    jitted = jit_fn(tree)

    assert jnp.array_equal(eager["value"], jitted["value"])


# ── T-11: lax.scan composition ──────────────────────────────────────
def test_lax_scan_body_traces_user_fn_same_count_as_single_call():
    """``lax.scan(body, ..., length=N)`` traces the body **once** — not N
    times. We verify by comparing trace counts: a single direct call to
    the sweep equals a 100-iteration scan."""
    topo = symmetric_topology(depth=3, degree=2)
    tree = Tree.empty(topo, {"value": (2,)}).at[topo.is_leaf].set(value=jnp.ones((8, 2))
    )

    # Direct call: trace count = number of levels touched (= depth)
    direct_counter = [0]

    @up(reads_children=("value",), writes=("value",))
    def avg_direct(node, children, params):
        direct_counter[0] += 1
        return {"value": children.value.mean(0)}

    avg_direct(tree)["value"].block_until_ready()
    direct_count = direct_counter[0]

    # Long scan: same depth, traced once. (Fresh SweepFn ⇒ fresh compile.)
    scan_counter = [0]

    @up(reads_children=("value",), writes=("value",))
    def avg_scan(node, children, params):
        scan_counter[0] += 1
        return {"value": children.value.mean(0)}

    def body(carry, _):
        return avg_scan(carry), None

    final, _ = jax.lax.scan(body, tree, xs=None, length=100)
    final["value"].block_until_ready()

    assert scan_counter[0] == direct_count, (
        f"Direct call traced user fn {direct_count} times; "
        f"100-iteration lax.scan traced it {scan_counter[0]} times. "
        f"They must match — lax.scan compiles the body once, not per iter."
    )


def test_lax_scan_top_level_jaxpr_has_one_scan_primitive():
    """Structural check: the jaxpr of ``lax.scan(body, length=N)(tree)`` has
    a single top-level ``scan`` equation. If the body were re-traced or
    unrolled per iteration, we'd see many more equations."""
    topo = symmetric_topology(depth=3, degree=2)
    tree = Tree.empty(topo, {"value": (2,)}).at[topo.is_leaf].set(value=jnp.ones((8, 2))
    )

    @up(reads_children=("value",), writes=("value",))
    def avg(node, children, params):
        return {"value": children.value.mean(0)}

    def body(carry, _):
        return avg(carry), None

    def run_scan(t):
        final, _ = jax.lax.scan(body, t, xs=None, length=50)
        return final

    j = jax.make_jaxpr(run_scan)(tree)
    eqn_names = [str(eq.primitive) for eq in j.jaxpr.eqns]
    scan_count = sum(1 for name in eqn_names if name == "scan")
    assert scan_count == 1, (
        f"Expected exactly one top-level scan primitive; got {scan_count}. "
        f"Equations: {eqn_names}"
    )


def test_lax_scan_yields_numerically_correct_repeated_application():
    """Running an up-sweep K times via scan equals K direct applications."""
    topo = symmetric_topology(depth=3, degree=2)
    tree = Tree.empty(topo, {"value": ()}).at[topo.is_leaf].set(value=jnp.arange(8, dtype=jnp.float32)
    )

    @up(reads_children=("value",), writes=("value",))
    def avg(node, children, params):
        return {"value": children.value.mean(0)}

    # Eager loop
    eager = tree
    for _ in range(5):
        eager = avg(eager)

    # Scan
    def body(carry, _):
        return avg(carry), None
    scanned, _ = jax.lax.scan(body, tree, xs=None, length=5)

    assert jnp.array_equal(eager["value"], scanned["value"])


def test_lax_scan_works_on_unequal_degree_tree():
    """The segment-reduction path also composes with lax.scan."""
    topo = Topology.from_parents([0, 0, 0, 1, 1, 1, 2, 2])
    tree = Tree.empty(topo, {"value": ()}).at[topo.is_leaf].set(value=jnp.ones(5)
    )

    @up(reads_children=("value",), writes=("value",))
    def sum_up(node, children, params):
        return {"value": children.value.sum(0)}

    def body(carry, _):
        return sum_up(carry), None

    final, _ = jax.lax.scan(body, tree, xs=None, length=3)
    assert jnp.all(jnp.isfinite(final["value"]))


# ── jax.grad through sweeps ─────────────────────────────────────────
def test_grad_through_up_sweep_matches_hand_derivative():
    """``avg(leaves)`` averaged up to root: ``loss = root**2``.
    ``d_loss/d_leaf[i] = 2 * root / N_leaves`` for a symmetric binary tree."""
    topo = symmetric_topology(depth=2, degree=2)  # 4 leaves
    n_leaves = int(topo.is_leaf.sum())

    @up(reads_children=("value",), writes=("value",))
    def avg(node, children, params):
        return {"value": children.value.mean(0)}

    def loss_fn(leaf_vals):
        tree = Tree.empty(topo, {"value": ()}).at[topo.is_leaf].set(value=leaf_vals)
        return avg(tree)["value"][0] ** 2

    leaf_vals = jnp.ones(n_leaves)
    grads = jax.grad(loss_fn)(leaf_vals)
    # root = mean(leaves) = 1; d_loss/d_leaf = 2 * 1 / 4 = 0.5
    assert jnp.allclose(grads, 0.5)


def test_grad_through_down_sweep_matches_hand_derivative():
    """``propagate``: each node = parent + node.delta. Loss = sum of leaf
    values squared. Differentiating w.r.t. root.value should give
    ``2 * sum_leaf_value = 2 * (n_leaves * (root + sum_path_deltas))``."""
    topo = symmetric_topology(depth=2, degree=2)  # depth 2, 4 leaves
    # All deltas zero; leaves = root.
    tree = (
        Tree.empty(topo, {"value": (), "delta": ()})
        .set(delta=jnp.zeros(7))
    )

    @down(reads=("delta",), reads_parent=("value",), writes=("value",))
    def propagate(node, parent, params):
        return {"value": parent.value + node.delta}

    def loss_fn(root_val):
        t = tree.at[topo.is_root].set(value=jnp.array([root_val]))
        out = propagate(t)
        return (out["value"][topo.is_leaf] ** 2).sum()

    g = jax.grad(loss_fn)(3.0)
    # Each leaf = root = 3; loss = 4 * 9 = 36; d/d_root = 2 * (n_leaves * root) = 24
    assert jnp.allclose(g, 24.0)


def test_value_and_grad_through_unequal_degree_sweep():
    """Same end-to-end smoothness on the ragged segment_sum path."""
    topo = Topology.from_parents([0, 0, 0, 1, 1, 1, 2, 2])

    @up(reads_children=("value",), writes=("value",))
    def sum_up(node, children, params):
        return {"value": children.value.sum(0)}

    def loss_fn(leaf_vals):
        t = Tree.empty(topo, {"value": ()}).at[topo.is_leaf].set(value=leaf_vals)
        return sum_up(t)["value"][0] ** 2  # root squared

    leaf_vals = jnp.ones(5)
    v, g = jax.value_and_grad(loss_fn)(leaf_vals)
    # Root accumulates ALL 5 leaves via two-level sum: root = 5
    # d_root/d_leaf[i] = 1 for each leaf; d_loss/d_leaf[i] = 2 * root * 1 = 10
    assert jnp.allclose(v, 25.0)
    assert jnp.allclose(g, 10.0)


# ── Tree.update + sweep + jit ───────────────────────────────────────
def test_tree_update_then_sweep_uses_new_field():
    """Common pattern: build a Tree, add a brand-new field via .update,
    run a sweep that reads it. update returns a new Tree with extended
    schema; the sweep dispatches on the new pytree structure cleanly."""
    topo = symmetric_topology(depth=2, degree=2)
    tree = Tree.empty(topo, {"value": ()}).at[topo.is_leaf].set(value=jnp.ones(4))
    tree = tree.update(weights=jnp.arange(7, dtype=jnp.float32))

    @up(reads=("weights",), reads_children=("value",), writes=("value",))
    def weighted_sum(node, children, params):
        return {"value": children.value.sum(0) * node.weights}

    out = weighted_sum(tree)
    # Node 1: sum(1, 1) * weights[1]=1 → 2
    # Node 2: sum(1, 1) * weights[2]=2 → 4
    # Node 0: sum(node1=2, node2=4) * weights[0]=0 → 0
    assert jnp.allclose(out["value"][1], 2.0)
    assert jnp.allclose(out["value"][2], 4.0)
    assert jnp.allclose(out["value"][0], 0.0)


def test_tree_update_then_sweep_under_outer_jit():
    """Same as above, wrapped in @jax.jit."""
    topo = symmetric_topology(depth=2, degree=2)

    @up(reads=("weights",), reads_children=("value",), writes=("value",))
    def weighted_sum(node, children, params):
        return {"value": children.value.sum(0) * node.weights}

    @jax.jit
    def step(tree):
        return weighted_sum(tree)

    tree = Tree.empty(topo, {"value": ()}).at[topo.is_leaf].set(value=jnp.ones(4))
    tree = tree.update(weights=jnp.arange(7, dtype=jnp.float32))

    out = step(tree)
    assert jnp.allclose(out["value"][1], 2.0)
    assert jnp.allclose(out["value"][2], 4.0)
    assert jnp.allclose(out["value"][0], 0.0)


# ── 'train_step' style integration ──────────────────────────────────
def test_flax_style_train_step_pattern():
    """The exact pattern that broke the legacy code: an outer jit wrapping
    a sweep PLUS some external state (here, a simple loss + gradient).
    Old hyperiax leaked tracers; new hyperiax handles this naturally."""
    topo = symmetric_topology(depth=2, degree=2)

    @up(reads_children=("value",), writes=("value",))
    def avg(node, children, params):
        return {"value": children.value.mean(0)}

    def loss_and_grad(params, leaf_vals):
        def loss(p, lv):
            t = Tree.empty(topo, {"value": ()}).at[topo.is_leaf].set(value=lv)
            t = avg(t)
            return (t["value"][0] - p["target"]) ** 2
        return jax.value_and_grad(loss)(params, leaf_vals)

    @jax.jit
    def train_step(params, leaf_vals):
        loss, g = loss_and_grad(params, leaf_vals)
        new_params = {"target": params["target"] - 0.1 * g["target"]}
        return new_params, loss

    params = {"target": jnp.array(0.0)}
    leaf_vals = jnp.array([1.0, 2.0, 3.0, 4.0])

    for _ in range(10):
        params, _loss = train_step(params, leaf_vals)

    # Without leaked tracers and with correct grads, ``target`` should drift
    # toward the leaf mean (2.5).
    assert 1.0 < float(params["target"]) < 3.0
