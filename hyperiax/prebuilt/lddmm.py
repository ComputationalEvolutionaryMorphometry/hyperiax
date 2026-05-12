"""LDDMM-style landmark dynamics: drift / diffusion-covariance builders.

In LDDMM (Large Deformation Diffeomorphic Metric Mapping) for landmark
data, the state is ``n`` landmarks in ``R^d``, flattened to ``(n·d,)``.
The dynamics are free Brownian motion with a *spatially correlated*
diffusion: the covariance between landmarks is the kernel Gram matrix
``K_{ij} = k(x_i - x_j)``, applied tensor-product to ``I_d`` along the
spatial axis. The :mod:`hyperiax.prebuilt.sde` ``dot`` / ``solve``
helpers already operate on this tensor-product layout.

These constructors aren't in the legacy ``examples/shape.py`` — that
file only carried the kernels themselves. The drift and covariance
were inlined in research notebooks. We collect them here for re-use.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp


def lddmm_drift() -> Callable:
    """Zero-drift LDDMM dynamics: free Brownian motion in landmark space.

    Returns a function ``b(t, x, params) -> jnp.zeros_like(x)`` matching
    the signature expected by :func:`hyperiax.prebuilt.sde.forward`.
    """
    def b(t, x, params):
        return jnp.zeros_like(x)
    return b


def lddmm_covariance(kernel: Callable, *, n: int, d: int) -> Callable:
    """LDDMM diffusion covariance ``a(x, params) = K(x_i - x_j; params)``.

    The state ``x`` of shape ``(n·d,)`` is reshaped to ``(n, d)`` and
    pairwise differences ``(n, n, d)`` are fed to ``kernel``, which must
    return the Gram matrix of shape ``(n, n)``. Tensor-product structure
    means each spatial coordinate of each landmark shares the same Gram.

    Args:
        kernel: a callable from :mod:`hyperiax.prebuilt.shape_kernels`
            (or any ``(pairwise_diff, params) -> (n, n)``).
        n: number of landmarks.
        d: spatial dimension.
    """
    def a(x, params):
        landmarks = x.reshape((n, d))
        pairwise = landmarks[:, None, :] - landmarks[None, :, :]  # (n, n, d)
        return kernel(pairwise, params)
    return a
