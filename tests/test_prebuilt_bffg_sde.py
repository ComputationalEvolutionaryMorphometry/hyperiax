"""SDE BFFG (closed-form): backward_filter, forward_guided, and sweeps.

Cross-checks against :mod:`hyperiax.prebuilt.bffg_gaussian`: for free
Brownian motion (``b=0``, ``σ=I``, ``a=I``), the SDE BFFG math reduces
to the Gaussian closed form. Identical inputs must produce identical
``F_T`` and ``H_T``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hyperiax as hx
from hyperiax import Topology, Tree
from hyperiax.prebuilt import (
    gaussian_up,
    init_gaussian_leaves,
    init_sde_leaves,
    propagate_v_T_to_v_0,
    sde_down_conditional,
    sde_down_unconditional,
    sde_up,
)
from hyperiax.prebuilt.bffg_sde import backward_filter, forward_guided
from hyperiax.prebuilt.sde import dts


N, D = 1, 1
N_STEPS = 5


def _identity(v, params):
    return jnp.eye(N)


def _zero_drift(t, x, params):
    return jnp.zeros_like(x)


def _identity_sigma(x, params):
    return jnp.eye(N)


# ── pure-math: backward_filter against Gaussian closed form ────────
def test_backward_filter_brownian_matches_gaussian_closed_form():
    """Brownian motion has tildea = I everywhere → the SDE filter
    degenerates to the Gaussian closed-form update
    ``H_0 = H_T / (1 + H_T·T)``, ``F_0 = F_T / (1 + H_T·T)``."""
    T = 1.5
    H_T = jnp.array([[3.0]])
    F_T = jnp.array([2.0])
    v_T = jnp.array([2.0 / 3.0])  # = H_T⁻¹ F_T
    c_T = jnp.array([0.0])
    out = backward_filter(
        dts(T=T, n_steps=4),
        params={},
        c_T=c_T, v_T=v_T, F_T=F_T, H_T=H_T,
        tildea0=jnp.eye(1), tildeaT=jnp.eye(1),
    )
    expected_H = 3.0 / (1.0 + 3.0 * T)
    expected_F = 2.0 / (1.0 + 3.0 * T)
    assert jnp.allclose(out["H_0"][0, 0], expected_H, atol=1e-6)
    assert jnp.allclose(out["F_0"][0], expected_F, atol=1e-6)


# ── pure-math: forward_guided logpsi at the auxiliary endpoint ─────
def test_forward_guided_brownian_logpsi_is_zero():
    """When the auxiliary ``tildea`` matches the actual diffusion ``a``,
    the bridge-correction terms vanish: ``(a - tildea) = 0`` and the
    only remaining contribution is ``b · tilderx · dt = 0`` for ``b=0``."""
    T = 1.0
    n_steps = 20
    _dts = dts(T=T, n_steps=n_steps)
    dWs = jax.random.normal(jax.random.PRNGKey(0), (n_steps, 1))
    x0 = jnp.array([0.5])
    H_T = jnp.array([[2.0]])
    F_T = jnp.array([1.0])
    Xs, logpsi = forward_guided(
        x0, _dts, dWs, _zero_drift, _identity_sigma, params={},
        a=_identity, F_T=F_T, H_T=H_T,
        tildea0=jnp.eye(1), tildeaT=jnp.eye(1),
    )
    assert abs(float(logpsi)) < 1e-5
    assert Xs.shape == (n_steps + 1, 1)


def test_forward_guided_zero_noise_drives_state_toward_target():
    """With zero noise, the guided drift ``a·(F - H·X)·dt`` pushes ``X``
    toward ``H⁻¹·F`` (the conditional mean of the target Gaussian). At
    large T relative to ``H``, the trajectory should land near the target."""
    T = 5.0
    n_steps = 200
    _dts = dts(T=T, n_steps=n_steps)
    dWs = jnp.zeros((n_steps, 1))
    x0 = jnp.array([0.0])
    H_T = jnp.array([[1.0]])
    F_T = jnp.array([3.0])
    Xs, _ = forward_guided(
        x0, _dts, dWs, _zero_drift, _identity_sigma, params={},
        a=_identity, F_T=F_T, H_T=H_T,
        tildea0=jnp.eye(1), tildeaT=jnp.eye(1),
    )
    # The bridge target at time T is v_T = H⁻¹·F = 3.0; the guided
    # ODE relaxes toward it. Allow a generous slack.
    assert abs(float(Xs[-1, 0]) - 3.0) < 0.5


# ── sde_up vs gaussian_up on a Brownian tree ────────────────────────
def _make_sde_tree(edge_lengths, leaf_values, obs_var, root_value=0.0):
    """A 3-node tree with the SDE BFFG schema."""
    topo = Topology.from_parents([0, 0, 0])
    schema = {
        "edge_length": (),
        "value": (N_STEPS + 1, N * D),
        "noise": (N_STEPS, N * D),
        "c_T": (D,),
        "F_T": (N * D,),
        "H_T": (N, N),
        "v_T": (N * D,),
        "v_0": (N * D,),
        "logpsi": (),
    }
    tree = Tree.empty(topo, schema).set(edge_length=jnp.asarray(edge_lengths))
    tree = init_sde_leaves(
        tree,
        jnp.asarray(leaf_values),
        obs_var=obs_var,
        n=N, d=D,
        root_value=jnp.array([root_value]),
    )
    return tree, topo


def test_sde_up_brownian_matches_gaussian_up_bit_for_bit():
    edge_lengths = [0.0, 1.0, 2.0]
    leaf_values = [[1.0], [2.0]]
    obs_var = 0.1

    sde_tree, topo = _make_sde_tree(edge_lengths, leaf_values, obs_var)
    sde_out = sde_up(n_steps=N_STEPS, a=_identity)(sde_tree)

    gauss_tree = Tree.empty(topo, {
        "edge_length": (), "c_T": (D,), "F_T": (N * D,), "H_T": (N, N),
    }).set(edge_length=jnp.asarray(edge_lengths))
    gauss_tree = init_gaussian_leaves(
        gauss_tree, jnp.asarray(leaf_values), obs_var=obs_var, n=N, d=D
    )
    gauss_out = gaussian_up(n=N, a=_identity, d=D)(gauss_tree)

    np.testing.assert_allclose(
        np.asarray(sde_out["F_T"]), np.asarray(gauss_out["F_T"]), atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(sde_out["H_T"]), np.asarray(gauss_out["H_T"]), atol=1e-6
    )


def test_sde_up_root_posterior_matches_hand_formula():
    """Same closed form as the Gaussian case:
    H_root = 1/(τ²+l₁σ²) + 1/(τ²+l₂σ²); F_root = y₁/(τ²+l₁σ²) + y₂/(τ²+l₂σ²)."""
    edge_lengths = [0.0, 1.0, 2.0]
    obs_var = 0.1
    y1, y2 = 1.0, 2.0
    sde_tree, _ = _make_sde_tree(edge_lengths, [[y1], [y2]], obs_var)
    out = sde_up(n_steps=N_STEPS, a=_identity)(sde_tree)

    H = 1.0 / (obs_var + 1.0) + 1.0 / (obs_var + 2.0)
    F = y1 / (obs_var + 1.0) + y2 / (obs_var + 2.0)
    assert jnp.allclose(out["H_T"][0, 0, 0], H, atol=1e-5)
    assert jnp.allclose(out["F_T"][0, 0], F, atol=1e-5)
    assert jnp.allclose(out["v_T"][0, 0], F / H, atol=1e-5)


# ── propagate_v_T_to_v_0 sets the right linearization point ────────
def test_propagate_v_T_to_v_0_copies_parent_v_T():
    edge_lengths = [0.0, 1.0, 2.0]
    sde_tree, topo = _make_sde_tree(edge_lengths, [[1.0], [2.0]], obs_var=0.1)
    after_up = sde_up(n_steps=N_STEPS, a=_identity)(sde_tree)
    after_prop = propagate_v_T_to_v_0()(after_up)

    # Children of root (nodes 1, 2) should have v_0 == root's v_T
    root_v_T = after_up["v_T"][0]
    assert jnp.allclose(after_prop["v_0"][1], root_v_T)
    assert jnp.allclose(after_prop["v_0"][2], root_v_T)
    # Root v_0 is untouched by the down sweep.
    assert jnp.allclose(after_prop["v_0"][0], sde_tree["v_0"][0])


# ── sde_down_unconditional ──────────────────────────────────────────
def test_sde_down_unconditional_zero_noise_constant_trajectory():
    """Brownian (b=0) with zero noise: every leaf trajectory equals the
    parent's terminal state, replicated across all n_steps."""
    edge_lengths = [0.0, 1.0, 2.0]
    sde_tree, topo = _make_sde_tree(edge_lengths, [[1.0], [2.0]], obs_var=0.1)
    # Seed the root's trajectory with a known constant value.
    root_traj = jnp.full((1, N_STEPS + 1, N * D), 7.0)
    sde_tree = sde_tree.set_at(topo.is_root, value=root_traj)

    sweep = sde_down_unconditional(N_STEPS, _zero_drift, _identity_sigma)
    out = sweep(sde_tree)

    for node_idx in [1, 2]:
        # All time steps along this edge equal 7.0
        assert jnp.allclose(out["value"][node_idx], 7.0, atol=1e-5)


def test_sde_down_unconditional_noise_drives_diffusion():
    """With unit-σ Brownian noise, the leaf terminal value's variance
    should be ~ edge_length (variance accumulates linearly in time)."""
    edge_lengths = [0.0, 1.0]
    # Use a single-child topology for a clean variance estimate.
    topo = Topology.from_parents([0, 0])
    schema = {
        "edge_length": (),
        "value": (N_STEPS + 1, N * D),
        "noise": (N_STEPS, N * D),
    }
    sweep = sde_down_unconditional(N_STEPS, _zero_drift, _identity_sigma)

    def one_sample(key):
        tree = Tree.empty(topo, schema).set(edge_length=jnp.asarray(edge_lengths))
        # Root trajectory is all zeros (already zero from .empty).
        noise = jnp.zeros((topo.size, N_STEPS, N * D))
        noise = noise.at[1].set(jax.random.normal(key, (N_STEPS, N * D)))
        tree = tree.set(noise=noise)
        return sweep(tree)["value"][1, -1, 0]  # leaf terminal state

    keys = jax.random.split(jax.random.PRNGKey(0), 1000)
    terminal = jax.vmap(one_sample)(keys)
    var = float(jnp.var(terminal))
    # Expected variance ~ edge_length = 1.0
    assert abs(var - 1.0) < 0.25


# ── sde_down_conditional composes with sde_up ───────────────────────
def test_sde_up_then_down_conditional_pipeline_runs():
    """End-to-end: build a tree, run up sweep, propagate v_0, run conditional
    down. Must produce finite values for the trajectories and logpsi."""
    edge_lengths = [0.0, 1.0, 2.0]
    sde_tree, topo = _make_sde_tree(
        edge_lengths, [[1.0], [2.0]], obs_var=0.1, root_value=0.0,
    )
    sde_tree = sde_tree.set(noise=jnp.zeros((topo.size, N_STEPS, N * D)))

    after_up = sde_up(n_steps=N_STEPS, a=_identity)(sde_tree)
    after_prop = propagate_v_T_to_v_0()(after_up)

    cond = sde_down_conditional(N_STEPS, _zero_drift, _identity_sigma, _identity)
    out = cond(after_prop)
    assert jnp.all(jnp.isfinite(out["value"]))
    assert jnp.all(jnp.isfinite(out["logpsi"]))


# ── jit / scan compose ─────────────────────────────────────────────
def test_sde_up_composes_under_outer_jit():
    edge_lengths = [0.0, 1.0, 2.0]
    sde_tree, _ = _make_sde_tree(edge_lengths, [[1.0], [2.0]], obs_var=0.1)
    sweep = sde_up(n_steps=N_STEPS, a=_identity)
    eager = sweep(sde_tree)
    jitted = jax.jit(sweep)(sde_tree)
    assert jnp.allclose(eager["F_T"], jitted["F_T"])
    assert jnp.allclose(eager["H_T"], jitted["H_T"])


def test_sde_up_composes_under_lax_scan():
    edge_lengths = [0.0, 1.0, 2.0]
    sde_tree, _ = _make_sde_tree(edge_lengths, [[1.0], [2.0]], obs_var=0.1)
    sweep = sde_up(n_steps=N_STEPS, a=_identity)

    def body(carry, _):
        return sweep(carry), None

    final, _ = jax.lax.scan(body, sde_tree, xs=None, length=3)
    assert jnp.all(jnp.isfinite(final["F_T"]))


# ── ODE-integrated path ────────────────────────────────────────────
def _B_zero(t, params):
    return jnp.zeros((N, N))


def _beta_zero(t, params):
    return jnp.zeros(N)


def _B_damping(t, params):
    return -0.5 * jnp.eye(N)


def test_backward_filter_ode_with_zero_drift_matches_closed_form():
    """The ODE-integrated path with ``B = β = 0`` must reproduce the
    closed-form ``H_0 = H_T / (1 + H_T·T)`` etc. to numerical tolerance."""
    T = 1.5
    n_steps = 8
    _dts = dts(T=T, n_steps=n_steps)
    H_T = jnp.array([[2.0]])
    F_T = jnp.array([3.0])
    v_T = jnp.array([1.5])
    c_T = jnp.array([0.5])
    tildea = jnp.eye(1)

    cf = backward_filter(
        _dts, {}, c_T, v_T, F_T, H_T,
        tildea0=tildea, tildeaT=tildea,
    )
    ode = backward_filter(
        _dts, {}, c_T, v_T, F_T, H_T,
        tildea0=tildea, tildeaT=tildea, B=_B_zero, beta=_beta_zero,
    )

    # diffrax Tsit5 + PI controller (rtol=1e-7, atol=1e-9) is at least 1e-6 accurate.
    assert jnp.allclose(cf["H_0"], ode["H_0"], atol=1e-6)
    assert jnp.allclose(cf["F_0"], ode["F_0"], atol=1e-6)
    assert jnp.allclose(cf["c_0"], ode["c_0"], atol=1e-5)
    # ODE path additionally returns per-step series.
    assert ode["F_t"].shape == (n_steps, 1)
    assert ode["H_t"].shape == (n_steps, 1, 1)


def test_backward_filter_ode_returns_correct_endpoints():
    """The first per-step entry is the state at t = dt (end of step 0)
    and the last entry is at t = T (end of step n_steps-1, which is
    also where the integration started in t-time)."""
    T = 1.0
    n_steps = 4
    _dts = dts(T=T, n_steps=n_steps)
    H_T = jnp.array([[1.0]])
    F_T = jnp.array([2.0])
    v_T = jnp.array([2.0])
    c_T = jnp.array([0.0])

    ode = backward_filter(
        _dts, {}, c_T, v_T, F_T, H_T,
        tildea0=jnp.eye(1), tildeaT=jnp.eye(1),
        B=_B_zero, beta=_beta_zero,
    )
    # At t = T, H_t equals H_T (the initial condition of the backward sweep).
    assert jnp.allclose(ode["H_t"][-1], H_T, atol=1e-6)
    assert jnp.allclose(ode["F_t"][-1], F_T, atol=1e-6)


def test_forward_guided_ode_with_zero_drift_matches_closed_form():
    """Same inputs, same noise → trajectory and logpsi must match within
    the ODE integrator's tolerance."""
    T = 1.0
    n_steps = 16
    _dts = dts(T=T, n_steps=n_steps)
    H_T = jnp.array([[1.5]])
    F_T = jnp.array([2.0])
    x0 = jnp.array([0.3])
    dWs = jax.random.normal(jax.random.PRNGKey(0), (n_steps, 1))
    tildea = jnp.eye(1)

    Xs_cf, lp_cf = forward_guided(
        x0, _dts, dWs, _zero_drift, _identity_sigma, params={},
        a=_identity, F_T=F_T, H_T=H_T,
        tildea0=tildea, tildeaT=tildea,
    )

    filt = backward_filter(
        _dts, {}, jnp.zeros(1), jnp.zeros(1), F_T, H_T,
        tildea0=tildea, tildeaT=tildea, B=_B_zero, beta=_beta_zero,
    )
    Xs_ode, lp_ode = forward_guided(
        x0, _dts, dWs, _zero_drift, _identity_sigma, params={},
        a=_identity, F_t=filt["F_t"], H_t=filt["H_t"],
        tildea0=tildea, tildeaT=tildea,
        B=_B_zero, beta=_beta_zero,
    )
    assert jnp.allclose(Xs_cf, Xs_ode, atol=1e-5)
    assert jnp.allclose(lp_cf, lp_ode, atol=1e-5)


def test_sde_up_ode_with_zero_drift_matches_closed_form_sweep():
    edge_lengths = [0.0, 1.0, 2.0]
    sde_tree, _ = _make_sde_tree(
        edge_lengths, [[1.0], [2.0]], obs_var=0.1,
    )
    cf_out = sde_up(n_steps=N_STEPS, a=_identity)(sde_tree)
    ode_out = sde_up(
        n_steps=N_STEPS, a=_identity, B=_B_zero, beta=_beta_zero,
    )(sde_tree)
    assert jnp.allclose(cf_out["F_T"], ode_out["F_T"], atol=1e-5)
    assert jnp.allclose(cf_out["H_T"], ode_out["H_T"], atol=1e-5)
    assert jnp.allclose(cf_out["v_T"], ode_out["v_T"], atol=1e-5)


def test_sde_full_pipeline_with_nontrivial_damping_runs():
    """Damped OU-like drift (``B(t) = -0.5 I``): the full up → propagate →
    conditional-down pipeline must run and put each leaf's terminal
    state close to its observed value (with small slack from obs_var)."""
    edge_lengths = [0.0, 1.0, 1.0]
    sde_tree, topo = _make_sde_tree(
        edge_lengths, [[1.0], [-1.0]], obs_var=0.05,
        root_value=0.0,
    )
    sde_tree = sde_tree.set(noise=jnp.zeros((topo.size, N_STEPS, N * D)))

    up_sweep = sde_up(n_steps=N_STEPS, a=_identity, B=_B_damping, beta=_beta_zero)
    cond = sde_down_conditional(
        N_STEPS, _zero_drift, _identity_sigma, _identity,
        B=_B_damping, beta=_beta_zero,
    )
    t = up_sweep(sde_tree)
    t = propagate_v_T_to_v_0()(t)
    out = cond(t)
    assert jnp.all(jnp.isfinite(out["value"]))
    assert jnp.all(jnp.isfinite(out["logpsi"]))
    # Leaves land near their observations (within ~0.1 for obs_var=0.05).
    assert abs(float(out["value"][1, -1, 0]) - 1.0) < 0.1
    assert abs(float(out["value"][2, -1, 0]) - (-1.0)) < 0.1


def test_backward_filter_ode_without_diffrax_gives_clean_error(monkeypatch):
    """If the ``[prebuilt-bffg]`` extra isn't installed, the ODE path must
    raise an ImportError that points the user at the right extra."""
    import builtins
    import sys

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "diffrax":
            raise ImportError("simulated missing diffrax")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "diffrax", raising=False)

    from hyperiax.prebuilt.bffg_sde import _backward_filter_ode

    with pytest.raises(ImportError, match="prebuilt-bffg"):
        _backward_filter_ode(
            dts(T=1.0, n_steps=4), {},
            jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), jnp.eye(1),
            jnp.eye(1), jnp.eye(1),
            _B_zero, _beta_zero,
        )


def test_sde_up_ode_under_outer_jit():
    edge_lengths = [0.0, 1.0, 2.0]
    sde_tree, _ = _make_sde_tree(edge_lengths, [[1.0], [2.0]], obs_var=0.1)
    sweep = sde_up(n_steps=N_STEPS, a=_identity, B=_B_zero, beta=_beta_zero)
    eager = sweep(sde_tree)
    jitted = jax.jit(sweep)(sde_tree)
    assert jnp.allclose(eager["F_T"], jitted["F_T"], atol=1e-6)
    assert jnp.allclose(eager["H_T"], jitted["H_T"], atol=1e-6)
