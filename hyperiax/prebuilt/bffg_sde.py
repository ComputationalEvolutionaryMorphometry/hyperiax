"""Backward Filtering + Forward Guiding for SDE transitions on tree edges.

Each edge runs an SDE ``dx = b(t, x) dt + σ(x) dW`` for a duration equal to
the edge length, with the per-node value being the full trajectory of
shape ``(n_steps + 1, n·d)``. Leaves carry iid-Gaussian observations.
Inner-node posteriors are obtained by an up-sweep that runs a per-edge
:func:`backward_filter` and fuses ``(c, F, H)`` at the parent; conditional
forward sampling uses :func:`forward_guided` to draw bridges along each
edge.

This module covers BOTH the closed-form and the ODE-integrated BFFG
variants:

- **Closed form** (``B = β = None``): ``Φ_inv(t) = I + H_T · ∫ tildea du``
  has a clean analytic form and no ``odeint`` is needed. This is the
  common case for Brownian-like SDEs.
- **ODE-integrated** (``B``, ``β`` callables provided): the auxiliary
  linear drift ``tildeb(t, x) = β(t) + B(t)·x`` makes ``Φ_inv`` non-
  analytic, so ``(H, F, c)`` are integrated via
  :func:`jax.experimental.ode.odeint`. Assumes uniform time stepping
  (as produced by :func:`hyperiax.prebuilt.sde.dts`).

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
    tildea0: jax.Array | None,
    tildeaT: jax.Array,
    *,
    B=None,
    beta=None,
):
    """Backward filter over one SDE edge.

    Propagates unnormalized canonical Gaussian parameters ``(c, F, H)``
    backwards from the edge's lower end (subscript ``_T``, where the
    child sits) to its upper end (subscript ``_0``, where the parent
    sits).

    Two paths:

    - **Closed form** (``B is None and beta is None``): ``tildea(t)`` is
      linearly interpolated between ``tildea0`` and ``tildeaT``;
      ``Φ_inv(t) = I + H_T · ∫_t^T tildea(u) du`` has a clean analytic
      form. Returns ``{c_0, F_0, H_0}``.
    - **ODE path** (``B`` or ``beta`` callable provided): integrates
      ``(H, F, c)`` via :func:`jax.experimental.ode.odeint`. Returns
      ``{c_0, F_0, H_0, F_t, H_t}`` where ``F_t``/``H_t`` are the
      per-step series needed by :func:`forward_guided`.

    Args:
        dts: time discretization, shape ``(n_steps,)``. Assumed uniform
            for the ODE path. Total edge time is ``T = dts.sum()``.
        params: passed through to ``B``, ``beta`` (and the caller's ``a``).
        c_T, v_T, F_T, H_T: canonical Gaussian parameters at the lower end.
        tildea0, tildeaT: ``(n, n)`` auxiliary diffusivity at ``t=0`` and
            ``t=T``. ``tildea0=None`` defaults to constant ``tildeaT``.
        B: optional ``(t, params) -> (n, n)`` linear drift matrix.
        beta: optional ``(t, params) -> (n,)`` affine drift offset.
    """
    if tildea0 is None:
        tildea0 = tildeaT

    if B is None and beta is None:
        return _backward_filter_closed_form(
            dts, c_T, v_T, F_T, H_T, tildea0, tildeaT
        )
    return _backward_filter_ode(
        dts, params, c_T, v_T, F_T, H_T, tildea0, tildeaT, B, beta
    )


def _backward_filter_closed_form(dts, c_T, v_T, F_T, H_T, tildea0, tildeaT):
    n = tildeaT.shape[0]
    d = v_T.size // n

    if jnp.isscalar(c_T):
        c_T = jnp.array([c_T])

    T = dts.sum()
    # The linear closed-form Phi_inv comes from integrating
    # ``∫_0^T tildea(u) du`` = (T²/(2T))·tildeaT + (T²/(2T))·tildea0 / 2
    # … actually we want ∫_t^T tildea(u) du at t=0, which evaluates to
    #   (T/2)·tildeaT + (T/2)·tildea0 for the interpolated case and
    #   T·tildeaT in the constant case (both → (T/2)·(tildeaT+tildea0)
    #   when tildea0 = tildeaT).
    integrated = (T / 2.0) * tildeaT + (T / 2.0) * tildea0
    Phi_inv_0 = jnp.eye(n) + H_T @ integrated

    H_0 = jnp.linalg.solve(Phi_inv_0, H_T)
    F_0 = solve(Phi_inv_0, F_T)

    log_det_phi_inv = jnp.linalg.slogdet(Phi_inv_0)[1]
    c_0 = jax.vmap(
        lambda v_T_col, c_T_col: c_T_col
        + 0.5 * v_T_col @ (H_T - H_0) @ v_T_col
        - 0.5 * log_det_phi_inv,
        in_axes=(1, 0),
    )(v_T.reshape((n, d)), c_T)

    return {"c_0": c_0, "F_0": F_0, "H_0": H_0}


def _backward_filter_ode(dts, params, c_T, v_T, F_T, H_T, tildea0, tildeaT, B, beta):
    """ODE-integrated path for general linear ``B(t)``, ``β(t)``.

    Integrates ``(H, F, c)`` with τ = T - t (so τ ∈ [0, T] runs from
    the edge's lower end to its upper end). Returns ``H_t``, ``F_t``
    indexed by *step number* (``0..n_steps-1``), with ``H_t[i]`` /
    ``F_t[i]`` at the END of step ``i`` — matching :func:`forward_guided`'s
    expectations.
    """
    from jax.experimental.ode import odeint

    n = tildeaT.shape[0]
    d = v_T.size // n
    n_steps = dts.shape[0]
    T = dts.sum()

    nn = n * n
    nd = n * d

    if jnp.isscalar(c_T):
        c_T = jnp.array([c_T])

    def tildea(t):
        return tildeaT * (t / T) + tildea0 * (1 - t / T)

    def ode_system(y, tau, params_):
        # Recover (H, F, c) from the packed state.
        Ht = y[:nn].reshape((n, n))
        Ft_flat = y[nn:nn + nd]
        ct = y[nn + nd:]
        Ft_mat = Ft_flat.reshape((n, d))

        t = T - tau
        Bt = B(t, params_) if B is not None else jnp.zeros((n, n))
        betat = beta(t, params_) if beta is not None else jnp.zeros(n)
        ta = tildea(t)

        # Forward-in-t derivatives (van der Meulen et al. eqns).
        dH = -Bt.T @ Ht - Ht @ Bt + Ht @ ta @ Ht
        dF_mat = -Bt.T @ Ft_mat + Ht @ ta @ Ft_mat + (Ht @ betat)[:, None]
        dc = -(
            betat @ Ft_mat
            + 0.5 * jnp.einsum("id,ij,jd->d", Ft_mat, ta, Ft_mat)
            - 0.5 * jnp.trace(Ht @ ta) * jnp.ones(d)
        )

        # odeint integrates dy/dτ = f(y, τ); we want τ = T - t, so dy/dτ = -dy/dt.
        return -jnp.concatenate([dH.flatten(), dF_mat.flatten(), dc])

    y0 = jnp.concatenate([H_T.flatten(), F_T, c_T])
    # Uniform time stepping assumption: ts span [0, T] at n_steps+1 points.
    ts_tau = jnp.linspace(0.0, T, n_steps + 1)
    solution = odeint(ode_system, y0, ts_tau, params)
    # solution[k] = y at τ = ts_tau[k] = state at t = T - ts_tau[k].
    # Reverse so it's indexed by forward t from 0 to T.
    solution_t = solution[::-1]

    state_t0 = solution_t[0]
    per_step_end = solution_t[1:]  # length n_steps; H_t[i] at t = ts[i+1] = end of step i

    H_0 = state_t0[:nn].reshape((n, n))
    F_0 = state_t0[nn:nn + nd]
    c_0 = state_t0[nn + nd:]

    H_t = per_step_end[:, :nn].reshape((n_steps, n, n))
    F_t = per_step_end[:, nn:nn + nd]

    return {"c_0": c_0, "F_0": F_0, "H_0": H_0, "F_t": F_t, "H_t": H_t}


# ── pure-math: forward guided bridge (closed form) ─────────────────
def forward_guided(
    x0: jax.Array,
    dts: jax.Array,
    dWs: jax.Array,
    b,
    sigma,
    params,
    a,
    *,
    F_T: jax.Array | None = None,
    H_T: jax.Array | None = None,
    F_t: jax.Array | None = None,
    H_t: jax.Array | None = None,
    tildea0: jax.Array | None,
    tildeaT: jax.Array,
    B=None,
    beta=None,
):
    """Forward-guided SDE bridge.

    Integrates the *guided* SDE ``dx = b(t,x) dt + a(x)·(F(t) - H(t)·x) dt +
    σ(x) dW`` from ``x0`` over the edge. Two paths, mirroring
    :func:`backward_filter`:

    - **Closed form** (``B = β = None``, pass ``F_T``, ``H_T``): ``H(t)``
      and ``F(t)`` are recovered analytically via the linearly-interpolated
      Φ-inverse.
    - **ODE path** (``B``/``β`` callable, pass ``F_t``, ``H_t``): the
      per-step series from :func:`backward_filter` is indexed directly,
      and the auxiliary drift ``tildeb(t, x) = β(t) + B(t)·x`` enters
      the ``logpsi`` correction.

    Returns ``(Xs, logpsi)``.
    """
    if tildea0 is None:
        tildea0 = tildeaT

    if B is None and beta is None:
        assert F_T is not None and H_T is not None, (
            "Closed-form forward_guided needs F_T and H_T"
        )
        return _forward_guided_closed_form(
            x0, dts, dWs, b, sigma, params, a, F_T, H_T, tildea0, tildeaT
        )
    assert F_t is not None and H_t is not None, (
        "ODE forward_guided needs F_t and H_t (from backward_filter)"
    )
    return _forward_guided_ode(
        x0, dts, dWs, b, sigma, params, a, F_t, H_t, tildea0, tildeaT, B, beta
    )


def _forward_guided_closed_form(
    x0, dts, dWs, b, sigma, params, a, F_T, H_T, tildea0, tildeaT
):
    n = tildeaT.shape[0]
    d = x0.size // n
    ts = jnp.cumsum(dts)
    T = ts[-1]

    def tildea(t):
        return tildeaT * (t / T) + tildea0 * (1 - t / T)

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
        H_t_i = jnp.linalg.solve(Phi_inv_t, H_T)
        F_t_i = solve(Phi_inv_t, F_T)
        return _bridge_step_body(
            i, X, logpsi, dt, dW, t,
            H_t_i, F_t_i, b, sigma, a, params, tildea(t),
            tildeb=lambda tt, xx, pp: jnp.zeros_like(xx),
            n=n, d=d,
        )

    (_, X, logpsi), (_, Xs, _) = jax.lax.scan(step, (0, x0, 0.0), (dts, dWs))
    return jnp.vstack((Xs, X)), logpsi


def _forward_guided_ode(
    x0, dts, dWs, b, sigma, params, a, F_t_arr, H_t_arr, tildea0, tildeaT, B, beta
):
    n = tildeaT.shape[0]
    d = x0.size // n
    ts = jnp.cumsum(dts)
    T = ts[-1]

    def tildea(t):
        return tildeaT * (t / T) + tildea0 * (1 - t / T)

    def tildeb(t, X, params_):
        betat = beta(t, params_) if beta is not None else jnp.zeros(n)
        if B is None:
            return jnp.broadcast_to(betat[:, None], (n, d)).flatten()
        Bt = B(t, params_)
        return (Bt @ X.reshape((n, d)) + betat[:, None]).flatten()

    def step(carry, val):
        i, X, logpsi = carry
        dt, dW = val
        t = ts[i]
        H_t_i = H_t_arr[i]
        F_t_i = F_t_arr[i]
        return _bridge_step_body(
            i, X, logpsi, dt, dW, t,
            H_t_i, F_t_i, b, sigma, a, params, tildea(t),
            tildeb=tildeb,
            n=n, d=d,
        )

    (_, X, logpsi), (_, Xs, _) = jax.lax.scan(step, (0, x0, 0.0), (dts, dWs))
    return jnp.vstack((Xs, X)), logpsi


def _bridge_step_body(
    i, X, logpsi, dt, dW, t,
    H_t_i, F_t_i, b, sigma, a, params, tildea_t,
    *, tildeb, n, d,
):
    """Single Euler step of the guided SDE; shared between closed-form and ODE paths."""
    tilderx = F_t_i - dot(H_t_i, X)
    if sigma is not None:
        _sigma = sigma(X, params)
        _a = jnp.einsum("ij,kj->ik", _sigma, _sigma)
    else:
        _a = a(X, params)
        _sigma = cholesky(_a, lower=True, check_finite=False)

    Xtp1 = X + b(t, X, params) * dt + dot(_a, tilderx) * dt + dot(_sigma, dW)

    amtildea = _a - tildea_t
    drift_diff = b(t, X, params) - tildeb(t, X, params)
    logpsi_tp1 = logpsi + (
        jnp.dot(drift_diff, tilderx)
        - 0.5 * d * jnp.einsum("ij,ji->", amtildea, H_t_i)
        + 0.5
        * jnp.einsum(
            "ij,jd,id->",
            amtildea,
            tilderx.reshape((n, d)),
            tilderx.reshape((n, d)),
        )
    ) * dt
    return (i + 1, Xtp1, logpsi_tp1), (t, X, logpsi)


# ── up sweep (backward filter on each edge) ─────────────────────────
def sde_up(n_steps: int, a, *, B=None, beta=None) -> SweepFn:
    """SDE BFFG up-sweep (closed-form or ODE-integrated).

    Each parent's posterior canonical params ``(c_T, F_T, H_T)`` and
    posterior mean ``v_T`` are computed by:

    1. Per child, running :func:`backward_filter` over the edge to
       propagate child's ``(c_T, F_T, H_T)`` upward to ``(c_0, F_0, H_0)``.
    2. Summing the children's messages.
    3. Solving for the posterior mean ``v_T = H_T⁻¹ F_T``.

    Args:
        n_steps: time discretization for the per-edge filter. Closed-form
            doesn't actually need many steps — ``Φ_inv(0)`` is analytic
            — but the parameter is kept for API symmetry with the ODE
            path and the matching :func:`sde_down_conditional`.
        a: per-edge diffusion covariance ``(v, params) -> (n, n)``.
        B: optional ``(t, params) -> (n, n)``. If ``B`` or ``beta`` is
            provided the per-edge filter switches to the ODE path.
        beta: optional ``(t, params) -> (n,)``.

    Note: the ODE path produces ``F_t, H_t`` per edge that the matching
    conditional down sweep would otherwise need. We DON'T persist them
    on the tree; :func:`sde_down_conditional` re-runs the same ODE per
    edge during sampling. This is wasteful but keeps the up sweep's
    writes set minimal and the API symmetric with the closed-form path.
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
                B=B,
                beta=beta,
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
def sde_down_conditional(
    n_steps: int, b, sigma, a, *, B=None, beta=None,
) -> SweepFn:
    """Conditional forward sampling via :func:`forward_guided`.

    Closed-form when ``B = β = None``. With ``B``/``β`` provided, re-runs
    :func:`backward_filter` (ODE path) per edge to recover the ``F_t``,
    ``H_t`` series before invoking :func:`forward_guided`. (We don't
    persist ``F_t``/``H_t`` on the tree — see :func:`sde_up`.)

    Requires the up-sweep results ``v_0, v_T, c_T, F_T, H_T`` on each node
    (set them via :func:`init_sde_leaves` and :func:`sde_up`). Writes the
    bridge trajectory back to ``value`` and the cumulative log
    correction to ``logpsi``.
    """

    @down(
        reads=("noise", "edge_length", "v_0", "v_T", "c_T", "F_T", "H_T"),
        reads_parent=("value",),
        writes=("value", "logpsi"),
    )
    def _sweep(node, parent, params):
        var = node.edge_length
        _dts = dts(T=var, n_steps=n_steps)
        _dWs = jnp.sqrt(_dts)[:, None] * node.noise
        x0 = parent.value.reshape((n_steps + 1, -1))[-1]
        tildea0 = a(node.v_0, params)
        tildeaT = a(node.v_T, params)

        if B is None and beta is None:
            Xs, logpsi = forward_guided(
                x0, _dts, _dWs, b, sigma, params, a,
                F_T=node.F_T, H_T=node.H_T,
                tildea0=tildea0, tildeaT=tildeaT,
            )
        else:
            # Re-derive F_t, H_t for this edge via the ODE filter.
            filt = backward_filter(
                _dts, params, node.c_T, node.v_T, node.F_T, node.H_T,
                tildea0=tildea0, tildeaT=tildeaT,
                B=B, beta=beta,
            )
            Xs, logpsi = forward_guided(
                x0, _dts, _dWs, b, sigma, params, a,
                F_t=filt["F_t"], H_t=filt["H_t"],
                tildea0=tildea0, tildeaT=tildeaT,
                B=B, beta=beta,
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
