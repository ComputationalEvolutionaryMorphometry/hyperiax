"""SDE utilities: time discretization and Euler-Maruyama integration.

Pure-math helpers; no tree machinery. Ported from the legacy
``examples/SDE.py`` (signatures preserved).

The factorized :func:`dot` / :func:`solve` operate on a flattened
``(n·d,)`` state vector as if it were ``(n, d)`` — useful for
landmark-style data where each of ``n`` landmarks carries a
``d``-dimensional coordinate, and the diffusivity is a Kronecker
product ``A ⊗ I_d``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky


def dot(A: jax.Array, v: jax.Array) -> jax.Array:
    """Compute ``(A ⊗ I_d) @ v`` on a flattened state vector.

    ``v`` is interpreted as shape ``(n, d)`` where ``n = A.shape[0]``;
    the result is flattened back to ``(n·d,)``.
    """
    return jnp.einsum("ij,jd->id", A, v.reshape((A.shape[0], -1))).flatten()


def solve(A: jax.Array, v: jax.Array) -> jax.Array:
    """Solve ``(A ⊗ I_d) x = v`` for ``x``, flattened.

    Equivalent to :func:`dot` with ``A⁻¹``.
    """
    return jnp.linalg.solve(A, v.reshape((A.shape[0], -1))).flatten()


def dts(T: float = 1.0, n_steps: int = 100) -> jax.Array:
    """Uniform time discretization over ``[0, T]`` with ``n_steps`` intervals."""
    return jnp.array([T / n_steps] * n_steps)


def forward(x, dts, dWs, b, sigma, params, a=None) -> jax.Array:
    """Euler-Maruyama integration of ``dx = b(t, x) dt + σ(x) dW``.

    Args:
        x: initial state, shape ``(n·d,)``.
        dts: time increments, shape ``(n_steps,)``.
        dWs: Brownian increments, shape ``(n_steps, d)``.
        b: drift ``(t, x, params) -> (n·d,)``.
        sigma: diffusion factor ``(x, params) -> (n, n)`` such that
            ``σσᵀ = a``. Pass ``None`` and provide ``a`` to use a
            cholesky path instead.
        params: dict-like, passed through to ``b``, ``sigma``, and ``a``.
        a: optional diffusion covariance ``(x, params) -> (n, n)``;
            consulted only when ``sigma is None``.

    Returns:
        Trajectory ``Xs`` of shape ``(n_steps + 1, n·d)`` including the
        initial state at index 0.
    """
    def SDE(carry, val):
        t, X = carry
        dt, dW = val
        if sigma is not None:
            Xtp1 = X + b(t, X, params) * dt + dot(sigma(x, params), dW)
        else:
            assert a is not None, "either sigma or a must be provided"
            Xtp1 = X + b(t, X, params) * dt + dot(
                cholesky(a(x, params), lower=True, check_finite=False), dW
            )
        return ((t + dt, Xtp1), (t, X))

    (_, X), (_, Xs) = jax.lax.scan(SDE, (0.0, x), (dts, dWs))
    return jnp.vstack((Xs, X))
