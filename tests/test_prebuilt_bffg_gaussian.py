"""Gaussian BFFG up-sweep + unconditional down. Covers T-14 in spirit.

The legacy reference (`examples/ABFFG.py`) was removed in Stage 0, so the
parts of T-14 that required "same numerical output as the legacy code"
are now verified by hand-derived closed-form formulas instead. The math
is identical, by construction (the prebuilt is a re-packaging of the
legacy `Gaussian_up` / `Gaussian_down_unconditional` with the OOP
wrapper stripped out).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hyperiax as hx
from hyperiax import Topology, Tree, symmetric_topology
from hyperiax.prebuilt import (
    gaussian_down_conditional,
    gaussian_down_unconditional,
    gaussian_up,
    init_gaussian_leaves,
    phylo_mean,
)


def _make_3_node_tree(edge_lengths):
    """root + 2 leaves; scalar state (n=d=1)."""
    topo = Topology.from_parents([0, 0, 0])
    n, d = 1, 1
    tree = Tree.empty(topo, {
        "edge_length": (),
        "value": (n * d,),
        "noise": (n * d,),
        "c_T": (d,),
        "F_T": (n * d,),
        "H_T": (n, n),
    })
    return tree.set(edge_length=jnp.asarray(edge_lengths, dtype=jnp.float32))


# ── T-14: up sweep hand-computed posterior on a 3-node tree ────────
def test_gaussian_up_3_node_root_posterior_matches_closed_form():
    """For a root + 2 leaves with edge variances σ²·l_i and observation
    noise τ², the root canonical posterior is:

        H_root = 1/(τ² + l₁σ²) + 1/(τ² + l₂σ²)
        F_root = y₁/(τ² + l₁σ²) + y₂/(τ² + l₂σ²)
    """
    n, d = 1, 1
    sigma_sq = 1.5
    obs_var = 0.2
    l1, l2 = 1.0, 2.0
    y1, y2 = 1.0, 2.0

    tree = _make_3_node_tree([0.0, l1, l2])
    tree = init_gaussian_leaves(
        tree, jnp.array([[y1], [y2]]), obs_var=obs_var, n=n, d=d
    )

    def a(v, params):
        return params["sigma_sq"] * jnp.eye(n)

    out = gaussian_up(n, a, d=d)(tree, params={"sigma_sq": sigma_sq})

    H_expected = 1.0 / (obs_var + l1 * sigma_sq) + 1.0 / (obs_var + l2 * sigma_sq)
    F_expected = y1 / (obs_var + l1 * sigma_sq) + y2 / (obs_var + l2 * sigma_sq)

    assert jnp.allclose(out["H_T"][0].squeeze(), H_expected, atol=1e-5)
    assert jnp.allclose(out["F_T"][0].squeeze(), F_expected, atol=1e-5)


def test_gaussian_up_root_posterior_mean_matches_canonical_form():
    """Posterior mean = F_root / H_root for a scalar Gaussian."""
    n, d = 1, 1
    sigma_sq = 1.0
    obs_var = 0.01  # near-deterministic observations
    l1, l2 = 1.0, 2.0
    y1, y2 = 3.0, 5.0

    tree = _make_3_node_tree([0.0, l1, l2])
    tree = init_gaussian_leaves(tree, jnp.array([[y1], [y2]]), obs_var=obs_var, n=n, d=d)

    def a(v, params):
        return params["sigma_sq"] * jnp.eye(n)

    out = gaussian_up(n, a, d=d)(tree, params={"sigma_sq": sigma_sq})

    mean = float(out["F_T"][0].squeeze() / out["H_T"][0].squeeze())

    H = 1.0 / (obs_var + l1 * sigma_sq) + 1.0 / (obs_var + l2 * sigma_sq)
    F = y1 / (obs_var + l1 * sigma_sq) + y2 / (obs_var + l2 * sigma_sq)
    assert abs(mean - F / H) < 1e-5


# ── BFFG → phylo_mean limit (single-level, τ² → 0) ──────────────────
def test_gaussian_bffg_recovers_phylo_mean_for_star_tree_zero_obs_var():
    """On a star tree (root + leaves directly, no intermediate nodes),
    with τ² → 0 and σ² = 1, the BFFG root posterior mean equals the
    edge-length-weighted phylo_mean estimator.

    The equivalence is single-level only: for deeper trees BFFG and
    phylo_mean diverge — BFFG correctly propagates the residual posterior
    precision through each inner node, while phylo_mean treats each inner
    estimate as a fresh observation (no precision propagation)."""
    # 1 root + 5 leaves directly
    topo = Topology.from_parents([0, 0, 0, 0, 0, 0])
    n, d = 1, 1
    n_leaves = int(topo.is_leaf.sum())
    key = jax.random.PRNGKey(0)
    k_e, k_y = jax.random.split(key)
    edge_lengths = jax.random.uniform(k_e, (topo.size,), minval=0.3, maxval=2.0)
    leaf_vals = jax.random.normal(k_y, (n_leaves, n * d))

    bffg_tree = Tree.empty(topo, {
        "edge_length": (), "value": (n * d,), "noise": (n * d,),
        "c_T": (d,), "F_T": (n * d,), "H_T": (n, n),
    }).set(edge_length=edge_lengths)
    bffg_tree = init_gaussian_leaves(bffg_tree, leaf_vals, obs_var=1e-7, n=n, d=d)

    def a(v, params):
        return jnp.eye(n)  # σ² = 1

    out_bffg = gaussian_up(n, a, d=d)(bffg_tree)
    bffg_root_mean = float(
        (out_bffg["F_T"][0] / out_bffg["H_T"][0].squeeze()).squeeze()
    )

    pm_tree = Tree.empty(topo, {"estimated_value": (), "edge_length": ()})
    pm_tree = pm_tree.set(edge_length=edge_lengths)
    pm_tree = pm_tree.set_at(topo.is_leaf, estimated_value=leaf_vals.squeeze())
    pm_root = float(phylo_mean()(pm_tree)["estimated_value"][0])

    assert abs(bffg_root_mean - pm_root) < 1e-4, (
        f"BFFG root mean {bffg_root_mean} should match phylo_mean {pm_root} "
        f"on a star tree with τ²→0, σ²=1"
    )


# ── unconditional forward sampling ──────────────────────────────────
def test_gaussian_down_unconditional_no_noise_keeps_parent_value():
    topo = symmetric_topology(height=2, degree=2)
    n = 1
    tree = Tree.empty(topo, {
        "value": (n,), "noise": (n,), "edge_length": (),
    })
    tree = tree.set(edge_length=jnp.ones(topo.size))
    tree = tree.set_at(topo.is_root, value=jnp.array([[5.0]]))
    # noise = 0 → child value = parent value
    sweep = gaussian_down_unconditional(lambda v, p: jnp.eye(n))
    out = sweep(tree)
    assert jnp.all(out["value"] == 5.0)


def test_gaussian_down_unconditional_brownian_variance_at_leaves():
    """With sigma=I, root=0, all edges=1: leaf values are sums of three
    independent N(0, 1) increments → leaf variance ≈ 3 (depth) per coord."""
    depth = 3
    topo = symmetric_topology(height=depth, degree=2)
    n = 1
    tree = Tree.empty(topo, {
        "value": (n,), "noise": (n,), "edge_length": (),
    }).set(edge_length=jnp.ones(topo.size))

    sweep = gaussian_down_unconditional(lambda v, p: jnp.eye(n))

    def one_sample(key):
        noise = jax.random.normal(key, (topo.size, n))
        t = tree.set(noise=noise)
        return sweep(t)["value"][topo.is_leaf]  # (n_leaves, 1)

    keys = jax.random.split(jax.random.PRNGKey(0), 1000)
    leaf_samples = jax.vmap(one_sample)(keys).squeeze(-1)  # (1000, n_leaves)
    var = float(leaf_samples.var(0).mean())
    # Each leaf is a sum of `depth` independent N(0,1) increments → Var ~ depth
    assert abs(var - depth) / depth < 0.15  # 15% sampling tolerance


# ── outer-jit + scan compose ───────────────────────────────────────
def test_gaussian_up_under_outer_jit():
    n, d = 1, 1
    tree = _make_3_node_tree([0.0, 1.0, 2.0])
    tree = init_gaussian_leaves(tree, jnp.array([[1.0], [2.0]]), obs_var=0.1, n=n, d=d)

    def a(v, params):
        return jnp.eye(n)

    sweep = gaussian_up(n, a, d=d)
    eager = sweep(tree)
    jitted = jax.jit(sweep)(tree)
    assert jnp.allclose(eager["F_T"], jitted["F_T"])
    assert jnp.allclose(eager["H_T"], jitted["H_T"])


def test_gaussian_up_under_lax_scan():
    """Running gaussian_up as a scan body — the body is compiled once."""
    n, d = 1, 1
    tree = _make_3_node_tree([0.0, 1.0, 2.0])
    tree = init_gaussian_leaves(tree, jnp.array([[1.0], [2.0]]), obs_var=0.1, n=n, d=d)

    def a(v, params):
        return jnp.eye(n)

    sweep = gaussian_up(n, a, d=d)

    def body(carry, _):
        return sweep(carry), None

    final, _ = jax.lax.scan(body, sweep(tree), xs=None, length=5)
    assert jnp.all(jnp.isfinite(final["F_T"]))


# ── gaussian_down_conditional ───────────────────────────────────────
def _make_3_node_gaussian_conditional_tree(edge_lengths, leaf_values, obs_var):
    """Tree with the schema needed for full Gaussian BFFG (up + cond down)."""
    topo = Topology.from_parents([0, 0, 0])
    n, d = 1, 1
    tree = Tree.empty(topo, {
        "edge_length": (),
        "value": (n * d,),
        "noise": (n * d,),
        "c_T": (d,),
        "F_T": (n * d,),
        "H_T": (n, n),
        "logw": (),
    })
    tree = tree.set(edge_length=jnp.asarray(edge_lengths))
    tree = init_gaussian_leaves(
        tree, jnp.asarray(leaf_values), obs_var=obs_var, n=n, d=d,
    )
    return tree, topo


def test_gaussian_down_conditional_zero_noise_recovers_observations_in_low_noise_limit():
    """With ``obs_var → 0`` and zero noise: the conditional draw at each
    leaf reduces to the leaf observation (the bridge collapses to the
    deterministic posterior mean)."""
    n, d = 1, 1
    edge_lengths = [0.0, 1.0, 1.0]
    leaf_values = [[3.0], [5.0]]
    obs_var = 1e-3
    tree, topo = _make_3_node_gaussian_conditional_tree(
        edge_lengths, leaf_values, obs_var,
    )

    a = lambda v, p: jnp.eye(n)
    up_out = gaussian_up(n, a, d=d)(tree)

    # Seed the root with its posterior mean, then run the conditional down.
    root_mean = (up_out["F_T"][0] / up_out["H_T"][0, 0, 0]).reshape(1, n * d)
    t = up_out.set_at(topo.is_root, value=root_mean)
    t = t.set(noise=jnp.zeros((topo.size, n * d)))
    out = gaussian_down_conditional(n, a, d=d)(t)

    # Each leaf's sample should be near its observation (slack from finite obs_var).
    assert abs(float(out["value"][1, 0]) - 3.0) < 0.05
    assert abs(float(out["value"][2, 0]) - 5.0) < 0.05


def test_gaussian_down_conditional_zero_noise_is_deterministic():
    """Two runs with the same inputs (zero noise) must produce identical samples."""
    n, d = 1, 1
    tree, topo = _make_3_node_gaussian_conditional_tree(
        [0.0, 1.0, 2.0], [[1.0], [2.0]], obs_var=0.1,
    )
    a = lambda v, p: jnp.eye(n)
    up_out = gaussian_up(n, a, d=d)(tree)
    root_mean = (up_out["F_T"][0] / up_out["H_T"][0, 0, 0]).reshape(1, n * d)
    t = up_out.set_at(topo.is_root, value=root_mean)
    t = t.set(noise=jnp.zeros((topo.size, n * d)))
    sweep = gaussian_down_conditional(n, a, d=d)
    out1 = sweep(t)
    out2 = sweep(t)
    assert jnp.array_equal(out1["value"], out2["value"])
    assert jnp.array_equal(out1["logw"], out2["logw"])


def test_gaussian_down_conditional_pipeline_produces_finite_logw():
    n, d = 1, 1
    tree, topo = _make_3_node_gaussian_conditional_tree(
        [0.0, 1.0, 2.0], [[1.0], [2.0]], obs_var=0.1,
    )
    a = lambda v, p: jnp.eye(n)
    up_out = gaussian_up(n, a, d=d)(tree)
    root_mean = (up_out["F_T"][0] / up_out["H_T"][0, 0, 0]).reshape(1, n * d)
    t = up_out.set_at(topo.is_root, value=root_mean)
    # Random noise on each non-root node.
    t = t.set(noise=jax.random.normal(jax.random.PRNGKey(0), (topo.size, n * d)))
    out = gaussian_down_conditional(n, a, d=d)(t)
    assert jnp.all(jnp.isfinite(out["value"]))
    assert jnp.all(jnp.isfinite(out["logw"]))


def test_gaussian_down_conditional_under_outer_jit():
    n, d = 1, 1
    tree, topo = _make_3_node_gaussian_conditional_tree(
        [0.0, 1.0, 1.0], [[1.0], [-1.0]], obs_var=0.05,
    )
    a = lambda v, p: jnp.eye(n)
    up_out = gaussian_up(n, a, d=d)(tree)
    root_mean = (up_out["F_T"][0] / up_out["H_T"][0, 0, 0]).reshape(1, n * d)
    t = up_out.set_at(topo.is_root, value=root_mean)
    t = t.set(noise=jnp.zeros((topo.size, n * d)))
    sweep = gaussian_down_conditional(n, a, d=d)
    eager = sweep(t)
    jitted = jax.jit(sweep)(t)
    assert jnp.allclose(eager["value"], jitted["value"])
    assert jnp.allclose(eager["logw"], jitted["logw"])


def test_gaussian_up_then_down_conditional_lax_scan_pipeline():
    """Wrap up + conditional down in a single jit and run via lax.scan."""
    n, d = 1, 1
    tree, topo = _make_3_node_gaussian_conditional_tree(
        [0.0, 1.0, 1.0], [[1.0], [-1.0]], obs_var=0.1,
    )
    a = lambda v, p: jnp.eye(n)
    up_sweep = gaussian_up(n, a, d=d)
    down_sweep = gaussian_down_conditional(n, a, d=d)

    def step(carry, key):
        # Re-run up + conditional down each iteration with fresh noise.
        t_up = up_sweep(carry)
        root_mean = (t_up["F_T"][0] / t_up["H_T"][0, 0, 0]).reshape(1, n * d)
        t_seed = t_up.set_at(topo.is_root, value=root_mean)
        noise = jax.random.normal(key, (topo.size, n * d))
        t_seed = t_seed.set(noise=noise)
        return down_sweep(t_seed), t_seed["value"]

    keys = jax.random.split(jax.random.PRNGKey(0), 5)
    final, _ = jax.lax.scan(step, tree, keys)
    assert jnp.all(jnp.isfinite(final["value"]))
    assert jnp.all(jnp.isfinite(final["logw"]))
