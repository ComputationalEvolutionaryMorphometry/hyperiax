"""Laplace + Gaussian kernels for LDDMM and shape utilities.

Each kernel takes a *pairwise-difference* tensor ``x`` of shape
``(n, n, d)`` (landmark ``i`` minus landmark ``j`` along each spatial
axis) and returns the ``(n, n)`` Gram matrix. ``params['k_alpha']`` is
the kernel amplitude and ``params['k_sigma']`` is the length scale.

Ported verbatim from the legacy ``examples/shape.py``; only the
top-level lambdas were rewritten as plain functions for clearer
introspection (and so ``jax.jit`` shows a proper qualname in tracebacks).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def r(x, params):
    """Scaled landmark-pair distance ``√(ε + Σ_d (x_d / σ)²)``.

    The small ``1e-7`` floor keeps the gradient finite at coincident
    landmarks (``r → 0``).
    """
    return jnp.sqrt(1e-7 + jnp.sum(jnp.square(x / params["k_sigma"]), 2))


def k_Gaussian(x, params):
    """Gaussian (RBF) kernel."""
    return params["k_alpha"] / 2 * jnp.exp(-0.5 * jnp.sum(jnp.square(x) / params["k_sigma"], 2))


def k_K0(x, params):
    """Laplace K_0 kernel: ``α · exp(-r)``."""
    r_ = r(x, params)
    return params["k_alpha"] * jnp.exp(-r_)


def g_K0(x, params):
    """``g`` corresponding to the Laplace K_0 kernel (c=(d+1)/2 with d=3)."""
    r_ = r(x, params)
    return params["k_alpha"] ** 2 * (1 + r_ + (1 / 3) * r_ ** 2) * jnp.exp(-r_)


def k_K1(x, params):
    """Laplace K_1 kernel: ``α · (1 + r) · exp(-r)``."""
    r_ = r(x, params)
    return params["k_alpha"] * (1 + r_) * jnp.exp(-r_)


def g_K1(x, params):
    """``g`` corresponding to the Laplace K_1 kernel (c=(d+3)/2 with d=3)."""
    r_ = r(x, params)
    return params["k_alpha"] ** 2 * (
        1 + r_ + (45 / 105) * r_ ** 2 + (10 / 105) * r_ ** 3 + (1 / 105) * r_ ** 4
    ) * jnp.exp(-r_)


def k_K2(x, params):
    """Laplace K_2 kernel: ``α · (1 + r + r²/3) · exp(-r)``."""
    r_ = r(x, params)
    return params["k_alpha"] * (1 + r_ + (1 / 3) * r_ ** 2) * jnp.exp(-r_)


def k_K3(x, params):
    """Laplace K_3 kernel."""
    r_ = r(x, params)
    return params["k_alpha"] * (1 + r_ + (2 / 5) * r_ ** 2 + (1 / 15) * r_ ** 3) * jnp.exp(-r_)


def k_K4(x, params):
    """Laplace K_4 kernel."""
    r_ = r(x, params)
    return params["k_alpha"] * (
        1 + r_ + (45 / 105) * r_ ** 2 + (10 / 105) * r_ ** 3 + (1 / 105) * r_ ** 4
    ) * jnp.exp(-r_)


# ── sphere utilities ────────────────────────────────────────────────
def fibonacci_sphere(samples: int = 1) -> jnp.ndarray:
    """Approximately uniform points on the unit sphere via the golden-spiral.

    Returns a ``(samples, 3)`` array. Useful as an initial landmark
    configuration for LDDMM-on-a-sphere experiments.
    """
    if samples < 2:
        raise ValueError(f"need at least 2 samples, got {samples}")
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle (radians)
    i = np.arange(samples)
    y = 1 - (i / (samples - 1)) * 2
    radius = np.sqrt(1 - y * y)
    theta = phi * i
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return jnp.asarray(np.stack([x, y, z], axis=-1))


def mesh_sphere(target_vertices: int = 100):
    """An icosphere with approximately ``target_vertices`` vertices.

    Returns a ``trimesh.Trimesh``. Requires the optional ``trimesh``
    dependency (extra ``[prebuilt-shape]``).
    """
    try:
        import trimesh
    except ImportError as e:
        raise ImportError(
            "mesh_sphere requires trimesh. Install via "
            "`uv sync --extra prebuilt-shape` or `pip install 'hyperiax[prebuilt-shape]'`."
        ) from e

    subdivisions = 0
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1)
    while len(sphere.vertices) < target_vertices:
        subdivisions += 1
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1)
    while len(sphere.vertices) > target_vertices and subdivisions > 0:
        subdivisions -= 1
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1)
    return sphere
