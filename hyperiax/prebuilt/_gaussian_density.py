"""Gaussian density helpers used by the BFFG prebuilts.

Three parameterizations show up in backward filtering:

- Standard:   ``N(x | μ, Σ)``      — :func:`logphi`
- Precision:  ``N(x | μ, H = Σ⁻¹)`` — :func:`logphi_H`
- Canonical:  ``N(x | F = Hμ, H)``  — :func:`logphi_can`
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def quadratic(x: jax.Array, H: jax.Array) -> jax.Array:
    """``xᵀ H x`` for a 1-D ``x`` and matrix ``H``."""
    return jnp.dot(x, jnp.dot(H, x))


def logphi(x: jax.Array, mu: jax.Array, Sigma: jax.Array) -> jax.Array:
    """Log Gaussian density in standard form ``N(μ, Σ)``."""
    return jax.scipy.stats.multivariate_normal.logpdf(x, mu, Sigma)


def logphi_H(x: jax.Array, mu: jax.Array, H: jax.Array) -> jax.Array:
    """Log Gaussian density with precision ``H = Σ⁻¹``."""
    logZ = 0.5 * (jnp.linalg.slogdet(H)[1] - jnp.log(2 * jnp.pi) * H.shape[0])
    return logZ - 0.5 * quadratic(x - mu, H)


def logphi_can(y: jax.Array, F: jax.Array, H: jax.Array) -> jax.Array:
    """Log Gaussian density in canonical form: ``F = Hμ``, precision ``H``."""
    return logphi_H(y, jnp.linalg.solve(H, F), H)


def canonical_leaf_messages(
    leaf_values: jax.Array,
    obs_var: float | jax.Array,
    *,
    n: int,
    d: int = 1,
) -> dict[str, jax.Array]:
    """Per-leaf ``(H_T, F_T, c_T)`` seeds for a BFFG up-sweep with iid Gaussian
    observations ``y_i ~ N(x_i, obs_var · I_n)``.

    Args:
        leaf_values: ``(n_leaves, n·d)`` observed leaf states.
        obs_var: scalar observation noise variance ``τ²``.
        n: latent state dimension.
        d: data dimension; the canonical message factors across the ``d``
            iid data slices.

    Returns:
        Dict with ``H_T`` ``(n_leaves, n, n)``, ``F_T`` ``(n_leaves, n·d)``,
        ``c_T`` ``(n_leaves, d)``.
    """
    n_leaves = leaf_values.shape[0]
    H_T_leaf = jnp.eye(n) / obs_var
    H_T = jnp.broadcast_to(H_T_leaf, (n_leaves, n, n))
    F_T = jax.vmap(lambda v: (H_T_leaf @ v.reshape((n, d))).flatten())(leaf_values)

    Sigma_leaf = obs_var * jnp.eye(n)
    c_T = jax.vmap(
        lambda v: jax.vmap(
            lambda vc: logphi(jnp.zeros(n), vc, Sigma_leaf)
        )(v.reshape((n, d)).T)  # iterate over the d data columns
    )(leaf_values)

    return {"H_T": H_T, "F_T": F_T, "c_T": c_T}
