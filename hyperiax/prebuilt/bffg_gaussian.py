"""Backward Filtering + Forward Guiding for Gaussian transitions.

A simplified-closed-form BFFG variant where every edge carries a
Gaussian transition ``N(x_child | x_parent, edge_length · a(x_parent))``
and every leaf carries an iid Gaussian observation
``y_i ~ N(x_i, obs_var · I)``. Inner-node posteriors are obtained by an
up-sweep that maintains ``(c_T, F_T, H_T)`` in the unnormalized
canonical Gaussian form::

    U(x | c, F, H) = exp(c - ½ xᵀHx + Fᵀx) .

The math is ported essentially verbatim from the legacy
``examples/ABFFG.py``; what's changed is the surface — instead of an
``UpLambdaReducer`` with a hand-rolled ``transform`` and an explicit
``reductions`` dict, the user gets a single :class:`SweepFn` produced
by :func:`gaussian_up`.

Currently equal-degree trees only (the up sweep follows the
``@hx.up``/``vmap`` per-parent path; the unequal-degree
``ChildrenAxis`` proxy doesn't yet expose the elementwise arithmetic
this needs).

References
----------
- Van der Meulen, Schauer et al., *Continuous-discrete smoothing of
  diffusions* (https://arxiv.org/abs/2010.03509)
- Mider et al., *Automatic Backward Filtering Forward Guiding*
  (https://arxiv.org/abs/2203.04155)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky

from ..core.sweep import SweepFn, down, up
from ..core.tree import Tree
from ._gaussian_density import logphi, logphi_H, logphi_can, logU
from .sde import dot, solve


# ── BFFG up sweep (backward filter) ─────────────────────────────────
def gaussian_up(n: int, a, d: int = 1) -> SweepFn:
    """Build the Gaussian BFFG up-sweep.

    Each parent's posterior parameters ``(c_T, F_T, H_T)`` are computed
    from its children's posteriors and the closed-form Gaussian
    transition along each edge.

    Args:
        n: dimension of the latent state matrix (e.g. ``n`` landmarks).
        a: callable ``(v, params) -> (n, n)`` returning the per-edge
            diffusion covariance scaling (so that the edge covariance
            is ``edge_length · a(v, params)``). For a homogeneous
            isotropic process, ``a = lambda v, p: p['variance'] * jnp.eye(n)``.
        d: data dimension (1 for scalar, 2/3 for landmarks). The state
            on each node has shape ``(n·d,)``.
    """

    @up(
        reads_children=("edge_length", "c_T", "F_T", "H_T"),
        writes=("c_T", "F_T", "H_T"),
    )
    def _sweep(node, children, params):
        # Per-parent under vmap:
        #   children.edge_length : (k,)
        #   children.c_T         : (k, d)
        #   children.F_T         : (k, n*d)
        #   children.H_T         : (k, n, n)

        def per_child(edge_length, F_T_child, H_T_child):
            var = edge_length
            v_T = solve(H_T_child, F_T_child)
            covar = var * a(v_T, params)
            invPhi_0 = jnp.eye(n) + H_T_child @ covar
            H_0 = jnp.linalg.solve(invPhi_0, H_T_child)
            F_0 = solve(invPhi_0, F_T_child)
            return F_0, H_0

        F_0s, H_0s = jax.vmap(per_child)(
            children.edge_length, children.F_T, children.H_T
        )
        F_T = F_0s.sum(0)
        H_T = H_0s.sum(0)
        # Recompute c_T at the parent from the fused (F_T, H_T); the
        # contributions of child_c_0 are absorbed into this normalization
        # (see van der Meulen et al. §3 for the derivation).
        c_T = jax.vmap(
            lambda FT_col: logphi_can(jnp.zeros(n), FT_col, H_T),
            in_axes=1,
        )(F_T.reshape((n, d)))
        return {"c_T": c_T, "F_T": F_T, "H_T": H_T}

    return _sweep


# ── leaf initialization helper ──────────────────────────────────────
def init_gaussian_leaves(
    tree: Tree,
    leaf_values: jax.Array,
    obs_var: float | jax.Array,
    *,
    n: int,
    d: int = 1,
) -> Tree:
    """Seed ``H_T``, ``F_T``, ``c_T`` at the leaves for a Gaussian up-sweep.

    Args:
        tree: a Tree with at least the fields ``H_T (n, n)``, ``F_T (n·d,)``,
            and ``c_T (d,)`` declared in its schema.
        leaf_values: array of shape ``(n_leaves, n·d)`` carrying the
            observed leaf states.
        obs_var: scalar observation noise variance ``τ²``. Per-leaf
            precision is ``I_n / τ²``.
        n: latent state dimension.
        d: data dimension.

    Returns:
        A new Tree with the leaf rows of ``H_T``, ``F_T``, ``c_T`` set.
    """
    leaf_mask = tree.topology.is_leaf
    n_leaves = int(leaf_mask.sum())

    H_T_leaf = jnp.eye(n) / obs_var
    H_T_leaves = jnp.broadcast_to(H_T_leaf, (n_leaves, n, n))

    F_T_leaves = jax.vmap(lambda v: dot(H_T_leaf, v))(leaf_values)

    Sigma_leaf = obs_var * jnp.eye(n)
    c_T_leaves = jax.vmap(
        lambda v: jax.vmap(
            lambda vc: logphi(jnp.zeros(n), vc, Sigma_leaf)
        )(v.reshape((n, d)).T)  # iterate over the d data columns
    )(leaf_values)

    return tree.set_at(
        leaf_mask,
        H_T=H_T_leaves,
        F_T=F_T_leaves,
        c_T=c_T_leaves,
    )


# ── conditional forward sampling (BFFG forward-guided) ─────────────
def gaussian_down_conditional(n: int, a, d: int = 1) -> SweepFn:
    """Conditional forward sampling for Gaussian BFFG.

    For each non-root node, draws a sample of its state given:

    1. its parent's current state (just sampled by the previous level
       of the down sweep), and
    2. the subtree posterior summarized by ``(c_T, F_T, H_T)`` at this
       node (set by :func:`gaussian_up`).

    Writes the sampled state to ``value`` and the per-edge log
    importance-weight contribution to ``logw``.

    The per-edge filter intermediates ``(c_0, F_0, H_0)`` are *not*
    persisted on the tree by :func:`gaussian_up`; this sweep re-derives
    them inline from each node's own ``(c_T, F_T, H_T)`` and edge length
    — the same math the up sweep used. This avoids extending the up
    sweep's writes set and the associated schema growth.

    Args:
        n: latent state dimension.
        a: ``(v, params) -> (n, n)`` diffusion covariance. Evaluated at
            the *parent's* value for sampling (so the bridge respects
            the current SDE state), and at this node's posterior mean
            for the per-edge filter linearization (matching the up sweep).
        d: data dimension.
    """

    @down(
        reads=("noise", "edge_length", "c_T", "F_T", "H_T"),
        reads_parent=("value",),
        writes=("value", "logw"),
    )
    def _sweep(node, parent, params):
        # ── Re-derive per-edge filter intermediates (c_0, F_0, H_0)
        # using the same math as gaussian_up: linearization at v_T (the
        # node's own posterior mean from the up sweep).
        H_T_c = node.H_T
        F_T_c = node.F_T
        c_T_c = node.c_T
        var = node.edge_length

        v_T = solve(H_T_c, F_T_c)                           # (n*d,)
        covar_lin = var * a(v_T, params)                    # (n, n)
        invPhi_0 = jnp.eye(n) + H_T_c @ covar_lin
        H_0 = jnp.linalg.solve(invPhi_0, H_T_c)
        F_0 = solve(invPhi_0, F_T_c)
        c_0 = jax.vmap(
            lambda v_T_col, c_T_col, _:
                c_T_col
                - logphi_H(jnp.zeros(n), v_T_col, H_T_c)
                + logphi_H(jnp.zeros(n), v_T_col, H_0),
            in_axes=(1, 0, 1),
        )(v_T.reshape((n, d)), c_T_c.reshape(d), F_T_c.reshape((n, d)))

        # ── Conditional Gaussian sample given parent state.
        # Sigma evaluated at parent's value (matches the legacy convention).
        x = parent.value
        Sigma = var * a(x, params)
        invH = jnp.linalg.solve(jnp.eye(n) + Sigma @ H_T_c, Sigma)
        mu = dot(invH, F_T_c + solve(Sigma, x))
        new_value = mu + dot(
            cholesky(invH, lower=True, check_finite=False), node.noise
        )

        # ── log importance weight (van der Meulen et al. §6).
        Sigma_T_post = invH
        v_T_post = dot(Sigma_T_post, F_T_c)
        inv_covar_Sigma_T = jnp.linalg.solve(
            jnp.eye(n) + H_T_c @ Sigma_T_post, H_T_c,
        )
        logw = jnp.sum(jax.vmap(
            lambda vTpost_col, x_col, c_0_col, F_0_col:
                logphi_H(vTpost_col, x_col, inv_covar_Sigma_T)
                - logU(x_col, c_0_col, F_0_col, H_0),
            in_axes=(1, 1, 0, 1),
        )(
            v_T_post.reshape((n, d)),
            x.reshape((n, d)),
            c_0.reshape(d),
            F_0.reshape((n, d)),
        ))

        return {"value": new_value, "logw": logw}

    return _sweep


# ── unconditional forward sampling ──────────────────────────────────
def gaussian_down_unconditional(sigma) -> SweepFn:
    """Sample child = parent + sqrt(edge_length) · σ(parent) · noise.

    The simplest forward Gaussian sweep: each non-root node's value is
    its parent's value plus a Gaussian increment whose variance scales
    with the edge length. Requires ``noise`` (pre-sampled standard
    Gaussian per node) on the tree's schema; the user controls
    randomness by writing ``noise`` before calling this sweep.

    Args:
        sigma: callable ``(parent_value, params) -> (n, n)`` giving the
            per-edge diffusion factor (so ``σσᵀ = a``).
    """

    @down(
        reads=("noise", "edge_length"),
        reads_parent=("value",),
        writes=("value",),
    )
    def _sweep(node, parent, params):
        var = node.edge_length
        return {
            "value": parent.value + jnp.sqrt(var) * dot(sigma(parent.value, params), node.noise),
        }

    return _sweep
