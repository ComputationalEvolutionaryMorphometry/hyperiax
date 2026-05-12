"""Gaussian density helpers used by the BFFG prebuilts.

Several parameterizations show up in backward filtering + forward guiding:

- Standard:    ``N(x | μ, Σ)``      — :func:`logphi`
- Precision:   ``N(x | μ, H = Σ⁻¹)`` — :func:`logphi_H`
- Canonical:   ``N(x | F = Hμ, H)``  — :func:`logphi_can`
- Unnormalized canonical: ``U(y | c, F, H) = exp(c - ½ yᵀHy + Fᵀy)`` —
  :func:`logU`

Lifted verbatim from the legacy ``examples/ABFFG.py``; isolated here so
:mod:`hyperiax.prebuilt.bffg_gaussian` (and the future ``bffg_sde``) can
share them.
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


def logomega(Sigma: jax.Array) -> jax.Array:
    """Log normalizing constant of ``N(0, Σ)`` in standard form."""
    return 0.5 * (
        -jnp.linalg.slogdet(Sigma)[1] - jnp.log(2 * jnp.pi) * Sigma.shape[0]
    )


def logomega_H(H: jax.Array) -> jax.Array:
    """Log normalizing constant of ``N(0, H⁻¹)`` in precision form."""
    return 0.5 * (
        jnp.linalg.slogdet(H)[1] - jnp.log(2 * jnp.pi) * H.shape[0]
    )


def logphi_H(x: jax.Array, mu: jax.Array, H: jax.Array) -> jax.Array:
    """Log Gaussian density with precision ``H = Σ⁻¹``."""
    return logomega_H(H) - 0.5 * quadratic(x - mu, H)


def logphi_can(y: jax.Array, F: jax.Array, H: jax.Array) -> jax.Array:
    """Log Gaussian density in canonical form: ``F = Hμ``, precision ``H``."""
    return logphi_H(y, jnp.linalg.solve(H, F), H)


def logU(y: jax.Array, c: jax.Array, F: jax.Array, H: jax.Array) -> jax.Array:
    """Unnormalized log Gaussian density: ``c - ½ yᵀHy + Fᵀy``."""
    return c - 0.5 * quadratic(y, H) + jnp.dot(F, y)
