"""Backward Filtering + Forward Guiding for SDE transitions on tree edges.

Each edge runs an SDE ``dx = b(t, x) dt + σ(x) dW`` for a duration equal to
the edge length, with the per-node value being the full trajectory of
shape ``(n_steps + 1, n·d)``. Leaves carry iid-Gaussian observations.
Inner-node posteriors are obtained by an up-sweep that runs a per-edge
:func:`backward_filter` and fuses ``(c, F, H)`` at the parent; conditional
forward sampling uses :func:`forward_guided` to draw bridges along each
edge.

Scope (this module is the closed-form half of BFFG):

- *Available*: ``B = β = 0`` (the standard zero-linear-drift case).
  ``backward_filter`` and ``forward_guided`` have analytic Φ-inverses and
  don't need ODE integration.
- *Deferred*: time-varying ``B(t)``, ``β(t)``. The ODE path through
  ``jax.experimental.ode.odeint`` is non-trivial to test numerically and
  rarely used in practice — port when needed.

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
from ._gaussian_density import logphi
from .sde import dot, dts, forward, solve


# ── pure-math: backward filter (closed form) ───────────────────────
def backward_filter(
    dts: jax.Array,
    params,
    c_T: jax.Array,
    v_T: jax.Array,
    F_T: jax.Array,
    H_T: jax.Array,
    tildea0: jax.Array,
    tildeaT: jax.Array,
):
    """Closed-form backward filter over one SDE edge.

    Propagates unnormalized canonical Gaussian parameters ``(c, F, H)``
    backwards from the edge's lower end (subscript ``_T``, where the
    child sits) to its upper end (subscript ``_0``, where the parent
    sits). The auxiliary diffusivity ``tildea(t)`` is linearly
    interpolated between ``tildea0`` and ``tildeaT``::

        tildea(t) = (t / T) · tildeaT + (1 - t / T) · tildea0

    Args:
        dts: time discretization for the edge, shape ``(n_steps,)``.
            Total edge time is ``T = dts.sum()``.
        params: passed through to other callers; unused in the closed form.
        c_T, v_T, F_T, H_T: canonical Gaussian parameters at the lower end.
        tildea0, tildeaT: ``(n, n)`` auxiliary diffusivity matrices at
            ``t=0`` and ``t=T``.

    Returns:
        ``dict`` with keys ``c_0``, ``F_0``, ``H_0`` — canonical params
        propagated to the upper end. The ``c`` correction uses the
        Φ-inverse log-determinant rather than ``logphi_H`` differences
        (matches the legacy ``examples/ABFFG.py``).
    """
    n = tildeaT.shape[0]
    d = v_T.size // n

    if jnp.isscalar(c_T):
        c_T = jnp.array([c_T])

    T = dts.sum()
    # The "linear" closed-form Phi_inv comes from integrating
    # ``int_t^T tildea(u) du`` = (-(t²-T²)/(2T))·tildeaT + ((T-t)²/(2T))·tildea0
    # and post-multiplying by H_T.
    integrated = (-(0.0 ** 2 - T ** 2) / (2 * T)) * tildeaT + (
        ((T - 0.0) ** 2) / (2 * T)
    ) * tildea0
    Phi_inv_0 = jnp.eye(n) + H_T @ integrated

    H_0 = jnp.linalg.solve(Phi_inv_0, H_T)
    F_0 = solve(Phi_inv_0, F_T)

    log_det_phi_inv = jnp.linalg.slogdet(Phi_inv_0)[1]
    # c_0 = c_T + ½·v_Tᵀ·(H_T - H_0)·v_T - ½·log|det Phi_inv(0)|, per d column
    c_0 = jax.vmap(
        lambda v_T_col, c_T_col: c_T_col
        + 0.5 * v_T_col @ (H_T - H_0) @ v_T_col
        - 0.5 * log_det_phi_inv,
        in_axes=(1, 0),
    )(v_T.reshape((n, d)), c_T)

    return {"c_0": c_0, "F_0": F_0, "H_0": H_0}


# ── pure-math: forward guided bridge (closed form) ─────────────────
def forward_guided(
    x0: jax.Array,
    dts: jax.Array,
    dWs: jax.Array,
    b,
    sigma,
    params,
    a,
    F_T: jax.Array,
    H_T: jax.Array,
    tildea0: jax.Array,
    tildeaT: jax.Array,
):
    """Closed-form forward-guided SDE bridge.

    Integrates the *guided* SDE ``dx = b(t,x) dt + a(x)·(F(t) - H(t)·x) dt +
    σ(x) dW`` from ``x0`` over the edge. ``H(t)`` and ``F(t)`` are the
    backward-filter parameters at each time step (closed-form via the
    linearly-interpolated Φ-inverse). Returns the trajectory and a
    cumulative ``logpsi`` correction needed for likelihood evaluation.

    See van der Meulen / Mider et al. for the SDE.
    """
    n = tildeaT.shape[0]
    d = x0.size // n

    ts = jnp.cumsum(dts)
    T = ts[-1]

    # tildea(t) = (t/T)·tildeaT + (1 - t/T)·tildea0
    def tildea(t):
        return tildeaT * (t / T) + tildea0 * (1 - t / T)

    # integrated_t = -(t²-T²)/(2T)·tildeaT + ((T-t)²/(2T))·tildea0
    def Phi_inv(t):
        integrated = (-(t ** 2 - T ** 2) / (2 * T)) * tildeaT + (
            ((T - t) ** 2) / (2 * T)
        ) * tildea0
        return jnp.eye(n) + H_T @ integrated

    def step(carry, val):
        i, X, logpsi = carry
        dt, dW = val
        t = ts[i]
        Phi_inv_t = Phi_inv(t)
        H_t = jnp.linalg.solve(Phi_inv_t, H_T)
        F_t = solve(Phi_inv_t, F_T)
        tilderx = F_t - dot(H_t, X)
        if sigma is not None:
            _sigma = sigma(X, params)
            _a = jnp.einsum("ij,kj->ik", _sigma, _sigma)
        else:
            _a = a(X, params)
            _sigma = cholesky(_a, lower=True, check_finite=False)

        Xtp1 = X + b(t, X, params) * dt + dot(_a, tilderx) * dt + dot(_sigma, dW)

        # logpsi update — see van der Meulen et al. §6 for the derivation.
        amtildea = _a - tildea(t)
        # tildeb = 0 in the closed-form path (no linear drift).
        logpsi_tp1 = logpsi + (
            jnp.dot(b(t, X, params), tilderx)
            - 0.5 * d * jnp.einsum("ij,ji->", amtildea, H_t)
            + 0.5
            * jnp.einsum(
                "ij,jd,id->",
                amtildea,
                tilderx.reshape((n, d)),
                tilderx.reshape((n, d)),
            )
        ) * dt
        return (i + 1, Xtp1, logpsi_tp1), (t, X, logpsi)

    (_, X, logpsi), (_, Xs, _) = jax.lax.scan(step, (0, x0, 0.0), (dts, dWs))
    Xs_full = jnp.vstack((Xs, X))
    return Xs_full, logpsi


# ── up sweep (backward filter on each edge) ─────────────────────────
def sde_up(n_steps: int, a) -> SweepFn:
    """SDE BFFG up-sweep (closed form).

    Each parent's posterior canonical params ``(c_T, F_T, H_T)`` and
    posterior mean ``v_T`` are computed by:

    1. Per child, running :func:`backward_filter` over the edge to
       propagate child's ``(c_T, F_T, H_T)`` upward to ``(c_0, F_0, H_0)``.
    2. Summing the children's messages.
    3. Solving for the posterior mean ``v_T = H_T⁻¹ F_T``.

    Args:
        n_steps: time discretization for the per-edge filter. (Closed-form
            doesn't actually need many steps — ``Phi_inv(0)`` is analytic
            — but the parameter is kept for API symmetry with the ODE path
            and for use by ``forward_guided`` later.)
        a: per-edge diffusion covariance ``(v, params) -> (n, n)``.
    """

    @up(
        reads_children=("edge_length", "v_0", "c_T", "v_T", "F_T", "H_T"),
        writes=("c_T", "v_T", "F_T", "H_T"),
    )
    def _sweep(node, children, params):
        def per_child(edge_length, v_0_c, c_T_c, v_T_c, F_T_c, H_T_c):
            T = edge_length
            _dts = dts(T=T, n_steps=n_steps)
            return backward_filter(
                _dts,
                params,
                c_T_c,
                v_T_c,
                F_T_c,
                H_T_c,
                tildea0=a(v_0_c, params),
                tildeaT=a(v_T_c, params),
            )

        msgs = jax.vmap(per_child)(
            children.edge_length,
            children.v_0,
            children.c_T,
            children.v_T,
            children.F_T,
            children.H_T,
        )
        c_T_new = msgs["c_0"].sum(0)
        F_T_new = msgs["F_0"].sum(0)
        H_T_new = msgs["H_0"].sum(0)
        v_T_new = solve(H_T_new, F_T_new)
        return {
            "c_T": c_T_new,
            "v_T": v_T_new,
            "F_T": F_T_new,
            "H_T": H_T_new,
        }

    return _sweep


# ── unconditional forward sampling ──────────────────────────────────
def sde_down_unconditional(n_steps: int, b, sigma, *, a=None) -> SweepFn:
    """Unconditional forward sampling: each edge integrates the SDE
    forward from the parent's terminal state, driven by ``noise``."""

    @down(
        reads=("noise", "edge_length"),
        reads_parent=("value",),
        writes=("value",),
    )
    def _sweep(node, parent, params):
        var = node.edge_length
        _dts = dts(T=var, n_steps=n_steps)
        _dWs = jnp.sqrt(_dts)[:, None] * node.noise
        # Parent's value is a full trajectory; start from its terminal state.
        x0 = parent.value.reshape((n_steps + 1, -1))[-1]
        return {"value": forward(x0, _dts, _dWs, b, sigma, params, a=a)}

    return _sweep


# ── conditional forward sampling (forward-guided bridge) ────────────
def sde_down_conditional(n_steps: int, b, sigma, a) -> SweepFn:
    """Conditional forward sampling via :func:`forward_guided`.

    Requires the up-sweep results ``v_0, v_T, F_T, H_T`` on each node
    (set them via :func:`init_sde_leaves` and :func:`sde_up`). Writes
    the bridge trajectory back to ``value`` and the cumulative log
    correction to ``logpsi``.
    """

    @down(
        reads=("noise", "edge_length", "v_0", "v_T", "F_T", "H_T"),
        reads_parent=("value",),
        writes=("value", "logpsi"),
    )
    def _sweep(node, parent, params):
        var = node.edge_length
        _dts = dts(T=var, n_steps=n_steps)
        _dWs = jnp.sqrt(_dts)[:, None] * node.noise
        x0 = parent.value.reshape((n_steps + 1, -1))[-1]
        Xs, logpsi = forward_guided(
            x0,
            _dts,
            _dWs,
            b,
            sigma,
            params,
            a,
            F_T=node.F_T,
            H_T=node.H_T,
            tildea0=a(node.v_0, params),
            tildeaT=a(node.v_T, params),
        )
        return {"value": Xs, "logpsi": logpsi}

    return _sweep


# ── v_0 propagation (down sweep: v_0 of node ← v_T of parent) ──────
def propagate_v_T_to_v_0() -> SweepFn:
    """Set each non-root node's ``v_0`` equal to its parent's ``v_T``.

    Run this after :func:`sde_up` so that the linearization point for
    ``tildea`` on each edge reflects the posterior mean at the parent.
    The root's ``v_0`` should be initialized separately (typically to the
    root prior value)."""

    @down(reads_parent=("v_T",), writes=("v_0",))
    def _sweep(node, parent, params):
        return {"v_0": parent.v_T}

    return _sweep


# ── leaf initialization helper ──────────────────────────────────────
def init_sde_leaves(
    tree: Tree,
    leaf_values: jax.Array,
    obs_var: float | jax.Array,
    *,
    n: int,
    d: int = 1,
    root_value: jax.Array | None = None,
) -> Tree:
    """Seed ``H_T``, ``F_T``, ``c_T``, ``v_T`` at the leaves and (optionally)
    ``v_0``, ``v_T`` everywhere from a single root linearization point.

    The user is responsible for declaring the matching fields on the
    tree's schema (typically ``edge_length, noise, value, c_T, F_T, H_T,
    v_T, v_0, logpsi`` plus whatever extras the application needs).
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
        )(v.reshape((n, d)).T)
    )(leaf_values)

    tree = tree.set_at(
        leaf_mask,
        H_T=H_T_leaves,
        F_T=F_T_leaves,
        c_T=c_T_leaves,
        v_T=leaf_values,
    )

    if root_value is not None:
        root_value = jnp.asarray(root_value)
        if "v_0" in tree.schema:
            tree = tree.set(
                v_0=jnp.broadcast_to(root_value, (tree.size, n * d))
            )
        if "v_T" in tree.schema:
            # Seed v_T everywhere with the root (will be overwritten at
            # leaves and at inner nodes during the up sweep).
            tree = tree.set_at(
                ~leaf_mask,
                v_T=jnp.broadcast_to(root_value, (tree.size - n_leaves, n * d)),
            )

    return tree
