"""SDE solvers in hyperiax.utils.sde (Euler-Maruyama + Milstein)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperiax.utils.sde import EulerMaruyama, Milstein, dot, dts, solve, solve_sde


def _ts(T, n_steps):
    return jnp.concatenate([jnp.zeros(1), jnp.cumsum(dts(T=T, n_steps=n_steps))])


# ── Kronecker helpers ──────────────────────────────────────────────
def test_dot_solve_round_trip_scalar():
    A = jnp.array([[2.0]])
    v = jnp.array([3.0])
    assert jnp.allclose(solve(A, dot(A, v)), v)


def test_dot_solve_round_trip_multi_dim():
    A = jnp.array([[2.0, 0.5], [0.5, 3.0]])
    v = jnp.array([1.0, 2.0, 3.0, 4.0])  # (n=2, d=2)
    assert jnp.allclose(solve(A, dot(A, v)), v, atol=1e-6)


def test_dts_uniform_sum_equals_T():
    d = dts(T=2.5, n_steps=50)
    assert d.shape == (50,)
    assert jnp.allclose(d.sum(), 2.5)


# ── Euler-Maruyama ─────────────────────────────────────────────────
def test_em_zero_noise_matches_deterministic_euler_on_ou():
    """Zero noise -> the SDE reduces to explicit Euler; for dx=-x dt we get
    x_{k+1} = x_k(1-dt)."""
    n_steps, T = 20, 1.0
    ts = _ts(T, n_steps)
    dWs = jnp.zeros((n_steps, 1))
    ys = solve_sde(
        lambda t, y, a: -y,
        lambda t, y, a: jnp.eye(1),
        jnp.array([1.0]),
        ts,
        dWs,
        solver=EulerMaruyama(),
    )
    dt = T / n_steps
    expected = jnp.array([(1.0 - dt) ** k for k in range(n_steps + 1)])
    np.testing.assert_allclose(np.asarray(ys).flatten(), np.asarray(expected), atol=1e-5)


def test_em_brownian_terminal_variance_scales_with_T():
    n_steps, T = 1000, 2.0
    ts = _ts(T, n_steps)
    step_dt = jnp.diff(ts)

    def one(key):
        dWs = jnp.sqrt(step_dt)[:, None] * jax.random.normal(key, (n_steps, 1))
        ys = solve_sde(
            lambda t, y, a: jnp.zeros_like(y),
            lambda t, y, a: jnp.eye(1),
            jnp.array([0.0]),
            ts,
            dWs,
            solver=EulerMaruyama(),
        )
        return ys[-1, 0]

    terminals = jax.vmap(one)(jax.random.split(jax.random.PRNGKey(0), 500))
    assert abs(float(jnp.var(terminals)) - T) < 0.3


# ── Milstein ───────────────────────────────────────────────────────
def test_milstein_equals_em_for_additive_noise():
    """State-independent diffusion -> ∂σ/∂y = 0 -> the Milstein correction
    vanishes and the two schemes coincide exactly."""
    n_steps, T = 15, 1.0
    ts = _ts(T, n_steps)
    dWs = 0.1 * jnp.ones((n_steps, 2))
    drift = lambda t, y, a: jnp.array([0.3, -0.2])
    diffusion = lambda t, y, a: jnp.array([[1.0, 0.0], [0.5, 2.0]])  # constant
    y0 = jnp.array([0.0, 0.0])
    em = solve_sde(drift, diffusion, y0, ts, dWs, solver=EulerMaruyama())
    mil = solve_sde(drift, diffusion, y0, ts, dWs, solver=Milstein())
    np.testing.assert_allclose(np.asarray(em), np.asarray(mil), atol=1e-6)


def test_milstein_single_step_beats_em_on_geometric_bm():
    """1D GBM dX = σX dW has exact X(dt) = X0·exp(-½σ²dt + σ dW). The
    Milstein step includes the −½σ²dt correction that EM misses, so a
    single Milstein step lands closer to the exact solution."""
    sigma, x0, dt, dW = 0.8, 1.0, 0.5, 0.3
    ts = jnp.array([0.0, dt])
    dWs = jnp.array([[dW]])
    drift = lambda t, y, a: jnp.zeros_like(y)
    diffusion = lambda t, y, a: (sigma * y).reshape(1, 1)  # σX, shape (state, noise)

    exact = x0 * np.exp(-0.5 * sigma**2 * dt + sigma * dW)
    em = float(solve_sde(drift, diffusion, jnp.array([x0]), ts, dWs, solver=EulerMaruyama())[-1, 0])
    mil = float(solve_sde(drift, diffusion, jnp.array([x0]), ts, dWs, solver=Milstein())[-1, 0])

    # Closed-form Milstein value for this scalar case.
    expected_mil = x0 + sigma * x0 * dW + 0.5 * sigma**2 * x0 * (dW**2 - dt)
    np.testing.assert_allclose(mil, expected_mil, atol=1e-6)
    assert abs(mil - exact) < abs(em - exact)


# ── jit / vmap composition ─────────────────────────────────────────
@pytest.mark.parametrize("solver", [EulerMaruyama(), Milstein()])
def test_solvers_jit_and_vmap_compose(solver):
    n_steps = 30
    ts = _ts(1.0, n_steps)
    drift = lambda t, y, a: -y
    diffusion = lambda t, y, a: 0.5 * jnp.eye(1) * jnp.exp(y)  # state-dependent

    @jax.jit
    def run(y0, dWs):
        return solve_sde(drift, diffusion, y0, ts, dWs, solver=solver)

    out = run(jnp.array([0.1]), jnp.zeros((n_steps, 1)))
    assert out.shape == (n_steps + 1, 1)

    batched = jax.vmap(lambda y0: run(y0, jnp.zeros((n_steps, 1))))(jnp.array([[0.1], [0.2]]))
    assert batched.shape == (2, n_steps + 1, 1)
