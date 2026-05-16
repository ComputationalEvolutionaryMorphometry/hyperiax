"""SDE utilities (Euler-Maruyama). Pure-math; no tree machinery."""

import jax
import jax.numpy as jnp
import numpy as np

from hyperiax.prebuilt.sde import dot, dts, forward, solve


# ── dot / solve are inverses on the factorized layout ──────────────
def test_dot_solve_round_trip_scalar():
    A = jnp.array([[2.0]])
    v = jnp.array([3.0])
    assert jnp.allclose(solve(A, dot(A, v)), v)


def test_dot_solve_round_trip_multi_dim():
    A = jnp.array([[2.0, 0.5], [0.5, 3.0]])
    v = jnp.array([1.0, 2.0, 3.0, 4.0])  # (n=2, d=2)
    assert jnp.allclose(solve(A, dot(A, v)), v, atol=1e-6)


# ── time discretization ────────────────────────────────────────────
def test_dts_uniform_sum_equals_T():
    d = dts(T=2.5, n_steps=50)
    assert d.shape == (50,)
    assert jnp.allclose(d.sum(), 2.5)


# ── forward Euler-Maruyama ─────────────────────────────────────────
def test_forward_zero_noise_matches_explicit_euler_on_ou():
    """With zero noise, the SDE reduces to a deterministic Euler step;
    for ``dx = -x dt`` we get x_{k+1} = x_k - x_k · dt."""
    n_steps = 20
    T = 1.0
    _dts = dts(T=T, n_steps=n_steps)
    _dWs = jnp.zeros((n_steps, 1))

    def b(t, x, p):
        return -x

    def sigma(x, p):
        return jnp.eye(1)

    Xs = forward(jnp.array([1.0]), _dts, _dWs, b, sigma, params={})
    # Explicit Euler reference
    dt = T / n_steps
    expected = jnp.array([(1.0 - dt) ** k for k in range(n_steps + 1)])
    np.testing.assert_allclose(np.asarray(Xs).flatten(), np.asarray(expected), atol=1e-5)


def test_forward_zero_drift_variance_scales_with_T():
    """Brownian motion with σ=I: variance at terminal time = T."""
    n_steps = 1000
    T = 2.0
    _dts = dts(T=T, n_steps=n_steps)

    def b(t, x, p):
        return jnp.zeros_like(x)

    def sigma(x, p):
        return jnp.eye(1)

    key = jax.random.PRNGKey(0)
    n_samples = 500
    keys = jax.random.split(key, n_samples)

    def one_sample(k):
        dWs = jnp.sqrt(_dts)[:, None] * jax.random.normal(k, (n_steps, 1))
        Xs = forward(jnp.array([0.0]), _dts, dWs, b, sigma, params={})
        return Xs[-1, 0]

    terminals = jax.vmap(one_sample)(keys)
    # Var ~ T = 2.0; allow 15% slack for finite-sample noise
    assert abs(float(jnp.var(terminals)) - T) < 0.3


def test_forward_uses_a_when_sigma_is_none():
    """The ``a`` branch (cholesky of a) and the ``sigma`` branch must
    agree numerically when ``a = σσᵀ``."""
    n_steps = 5
    T = 0.5
    _dts = dts(T=T, n_steps=n_steps)
    rng = jax.random.PRNGKey(1)
    dWs = jax.random.normal(rng, (n_steps, 2))

    def b(t, x, p):
        return jnp.zeros_like(x)

    # Must be lower-triangular for ``cholesky(σσᵀ, lower=True) == σ``.
    sigma_mat = jnp.array([[1.0, 0.0], [0.5, 2.0]])

    def sigma(x, p):
        return sigma_mat

    def a(x, p):
        return sigma_mat @ sigma_mat.T

    x0 = jnp.array([0.1, 0.2])
    Xs_sigma = forward(x0, _dts, dWs, b, sigma, params={})
    Xs_a = forward(x0, _dts, dWs, b, None, params={}, a=a)
    # cholesky(a) lower-triangular = sigma's cholesky form — same noise term.
    np.testing.assert_allclose(np.asarray(Xs_sigma), np.asarray(Xs_a), atol=1e-5)


def test_forward_jit_composes():
    """forward is jit'd through (uses lax.scan internally)."""
    n_steps = 50
    _dts = dts(T=1.0, n_steps=n_steps)

    def b(t, x, p):
        return -x

    def sigma(x, p):
        return jnp.eye(1)

    @jax.jit
    def run(x0, dWs):
        return forward(x0, _dts, dWs, b, sigma, {})

    Xs = run(jnp.array([1.0]), jnp.zeros((n_steps, 1)))
    assert Xs.shape == (n_steps + 1, 1)
