"""LDDMM drift / covariance constructors + SDE integration smoke."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperiax.prebuilt.lddmm import lddmm_covariance, lddmm_drift
from hyperiax.prebuilt.sde import dts, forward
from hyperiax.prebuilt.shape_kernels import k_Gaussian, k_K0, k_K1


PARAMS = {"k_alpha": 2.0, "k_sigma": 0.5}


# ── drift ───────────────────────────────────────────────────────────
def test_lddmm_drift_is_zero_everywhere():
    b = lddmm_drift()
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    assert jnp.all(b(0.0, x, {}) == 0)
    assert jnp.all(b(5.0, jnp.ones(10), {"any": "params"}) == 0)


# ── covariance ──────────────────────────────────────────────────────
def test_lddmm_covariance_returns_n_by_n_gram():
    n, d = 4, 3
    x = jnp.arange(n * d, dtype=jnp.float32)
    a = lddmm_covariance(k_K1, n=n, d=d)
    K = a(x, PARAMS)
    assert K.shape == (n, n)


def test_lddmm_covariance_is_symmetric():
    n, d = 5, 2
    landmarks = jax.random.normal(jax.random.PRNGKey(0), (n, d))
    x = landmarks.flatten()
    a = lddmm_covariance(k_K0, n=n, d=d)
    K = a(x, PARAMS)
    assert jnp.allclose(K, K.T, atol=1e-5)


def test_lddmm_covariance_is_positive_definite():
    """For well-separated landmarks the kernel Gram matrix is PD."""
    n, d = 4, 2
    landmarks = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    x = landmarks.flatten()
    a = lddmm_covariance(k_K1, n=n, d=d)
    K = a(x, PARAMS)
    eigs = jnp.linalg.eigvalsh(K)
    assert bool(jnp.all(eigs > 0)), f"eigs = {np.asarray(eigs)}"


def test_lddmm_covariance_diagonal_equals_amplitude():
    """K(x_i, x_i) = α for K_0 / K_1 / etc., since r=0 → kernel = α."""
    n, d = 3, 2
    landmarks = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.5]])
    x = landmarks.flatten()
    for k in (k_K0, k_K1):
        K = lddmm_covariance(k, n=n, d=d)(x, PARAMS)
        np.testing.assert_allclose(
            np.asarray(jnp.diag(K)), PARAMS["k_alpha"], atol=1e-3
        )


# ── compose with sde.forward ────────────────────────────────────────
# Note: in sde.forward, each dW step has shape (n·d,) — same as the state.
# The tensor-product `dot(A, dW)` reshapes it to (n, d), applies the (n, n)
# cholesky factor of the kernel Gram matrix, then flattens back.
def test_lddmm_forward_with_zero_noise_is_constant():
    """Free Brownian motion with zero noise: landmarks don't move."""
    n, d = 3, 2
    landmarks = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    x0 = landmarks.flatten()
    n_steps = 10
    _dts = dts(T=1.0, n_steps=n_steps)
    _dWs = jnp.zeros((n_steps, n * d))

    b = lddmm_drift()
    a = lddmm_covariance(k_K1, n=n, d=d)
    Xs = forward(x0, _dts, _dWs, b, sigma=None, params=PARAMS, a=a)
    for k in range(n_steps + 1):
        assert jnp.allclose(Xs[k], x0)


def test_lddmm_forward_landmarks_close_to_each_other_are_highly_correlated():
    """Two landmarks separated by a tiny offset (« k_sigma) are essentially
    coincident in kernel-space: their off-diagonal Gram entry is ≈ α, so
    the tensor-product noise drives both landmarks together. Trajectories
    should be highly correlated.

    (Exact coincidence makes the Gram rank-deficient, which trips the
    cholesky — so we use a small offset for numerical stability.)"""
    n, d = 2, 1
    x0 = jnp.array([0.0, 1e-3])  # tiny offset
    n_steps = 50
    _dts = dts(T=1.0, n_steps=n_steps)
    dWs = jax.random.normal(jax.random.PRNGKey(0), (n_steps, n * d))

    b = lddmm_drift()
    a = lddmm_covariance(k_K1, n=n, d=d)
    Xs = forward(x0, _dts, dWs, b, sigma=None, params=PARAMS, a=a)

    landmark_traj = np.asarray(Xs.reshape((n_steps + 1, n, d))[:, :, 0])
    # Pearson correlation between the two landmark trajectories
    corr = np.corrcoef(landmark_traj[:, 0], landmark_traj[:, 1])[0, 1]
    assert corr > 0.99, f"expected highly-correlated trajectories, got corr={corr}"


def test_lddmm_forward_under_jit():
    n, d = 3, 2
    landmarks = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    x0 = landmarks.flatten()
    n_steps = 5
    _dts = dts(T=0.5, n_steps=n_steps)
    dWs = jax.random.normal(jax.random.PRNGKey(1), (n_steps, n * d))

    b = lddmm_drift()
    a = lddmm_covariance(k_K1, n=n, d=d)

    @jax.jit
    def run(x):
        return forward(x, _dts, dWs, b, sigma=None, params=PARAMS, a=a)

    Xs = run(x0)
    assert Xs.shape == (n_steps + 1, n * d)
    assert jnp.all(jnp.isfinite(Xs))
