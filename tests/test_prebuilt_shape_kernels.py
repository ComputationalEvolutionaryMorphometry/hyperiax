"""Shape kernels (Laplace family) + sphere utilities."""

import jax.numpy as jnp
import numpy as np
import pytest

from hyperiax.prebuilt.shape_kernels import (
    fibonacci_sphere,
    g_K0,
    g_K1,
    k_Gaussian,
    k_K0,
    k_K1,
    k_K2,
    k_K3,
    k_K4,
    r,
)


PARAMS = {"k_alpha": 2.0, "k_sigma": 0.5}


def _pairwise(landmarks: jnp.ndarray) -> jnp.ndarray:
    """(n, d) → (n, n, d) pairwise differences."""
    return landmarks[:, None, :] - landmarks[None, :, :]


def test_r_is_zero_at_coincident_landmarks_up_to_epsilon():
    """The 1e-7 floor inside ``r`` makes r(0) ≈ √1e-7, not literally zero."""
    x = jnp.zeros((3, 3, 2))
    assert float(r(x, PARAMS)[0, 0]) < 1e-3


def test_kernels_at_coincident_landmarks_equal_amplitude():
    """At r ≈ 0: K_0, K_1, K_2, K_3, K_4 → α; Gaussian → α/2."""
    x = jnp.zeros((3, 3, 2))
    for k in (k_K0, k_K1, k_K2, k_K3, k_K4):
        assert jnp.allclose(k(x, PARAMS)[0, 0], PARAMS["k_alpha"], atol=1e-3)
    assert jnp.allclose(k_Gaussian(x, PARAMS)[0, 0], PARAMS["k_alpha"] / 2)


def test_kernels_symmetric_in_pairwise_arg():
    """K(x_i - x_j) = K(x_j - x_i) for radial kernels — Gram matrix
    of any kernel evaluated on real landmarks must be symmetric."""
    landmarks = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.5]])
    pw = _pairwise(landmarks)
    for k in (k_K0, k_K1, k_K2, k_K3, k_K4, k_Gaussian):
        G = k(pw, PARAMS)
        assert jnp.allclose(G, G.T), f"{k.__name__} Gram not symmetric"


def test_kernels_decay_with_distance():
    """Each kernel is monotonically decreasing in r for r > 0."""
    pw_near = jnp.zeros((1, 1, 2)).at[0, 0, 0].set(0.1)
    pw_far = jnp.zeros((1, 1, 2)).at[0, 0, 0].set(2.0)
    for k in (k_K0, k_K1, k_K2, k_K3, k_K4, k_Gaussian):
        assert float(k(pw_near, PARAMS)[0, 0]) > float(k(pw_far, PARAMS)[0, 0]), (
            f"{k.__name__} not decreasing with distance"
        )


def test_g_K0_matches_old_formula_at_specific_r():
    """g_K0(x) = α²·(1 + r + r²/3)·exp(-r). Hand-evaluated at r=1
    (which means x/σ = (1, 0, ...) in 1D), we get α²·(1 + 1 + 1/3)·e⁻¹."""
    # 1 landmark pair with offset = (k_sigma, 0) so r = 1
    pw = jnp.array([[[PARAMS["k_sigma"], 0.0]]])  # shape (1, 1, 2)
    expected = PARAMS["k_alpha"] ** 2 * (1 + 1 + 1 / 3) * jnp.exp(-1.0)
    assert jnp.allclose(g_K0(pw, PARAMS)[0, 0], expected, atol=1e-3)


# ── fibonacci sphere ───────────────────────────────────────────────
def test_fibonacci_sphere_returns_unit_vectors():
    pts = fibonacci_sphere(200)
    assert pts.shape == (200, 3)
    norms = np.asarray(jnp.linalg.norm(pts, axis=-1))
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_fibonacci_sphere_approximately_uniform():
    """Mean position of N points on a sphere should be near origin."""
    pts = np.asarray(fibonacci_sphere(500))
    assert np.linalg.norm(pts.mean(0)) < 0.05


def test_fibonacci_sphere_rejects_n_lt_2():
    with pytest.raises(ValueError):
        fibonacci_sphere(0)


# ── mesh_sphere (trimesh optional) ─────────────────────────────────
def test_mesh_sphere_with_trimesh_returns_target_vertices():
    trimesh = pytest.importorskip("trimesh")
    from hyperiax.prebuilt.shape_kernels import mesh_sphere

    sphere = mesh_sphere(target_vertices=42)
    assert isinstance(sphere, trimesh.Trimesh)
    # Icosphere subdivision counts: 12, 42, 162, ... so target=42 hits exactly.
    assert len(sphere.vertices) == 42


def test_mesh_sphere_without_trimesh_gives_clean_error(monkeypatch):
    """Hide trimesh from the lazy import and verify the error message
    points the user at the right extra."""
    import sys
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "trimesh":
            raise ImportError("simulated missing trimesh")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "trimesh", raising=False)

    from hyperiax.prebuilt.shape_kernels import mesh_sphere

    with pytest.raises(ImportError, match="prebuilt-shape"):
        mesh_sphere(target_vertices=12)
