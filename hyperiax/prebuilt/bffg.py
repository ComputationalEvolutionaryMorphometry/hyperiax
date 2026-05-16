"""Backward Filtering + Forward Guiding (Gaussian and SDE transitions).

Both variants maintain the unnormalized canonical form
``U(x | c, F, H) = exp(c − ½ xᵀHx + Fᵀx)`` for the backward message and
share the same leaf initialization. They differ only in the per-edge
transition:

- **Gaussian** (:func:`gaussian_up` / :func:`gaussian_down_conditional` /
  :func:`gaussian_down_unconditional`): each edge carries a closed-form
  Gaussian transition ``N(x_child | x_parent, edge_length · a(x_parent))``.
- **SDE** (:func:`sde_up` / :func:`sde_down_conditional` /
  :func:`sde_down_unconditional`): each edge runs an SDE
  ``dx = b(t, x) dt + σ(x) dW`` for ``edge_length`` time units; the
  per-node value is the full trajectory of shape ``(n_steps + 1, n·d)``.
  Closed-form when ``B = β = None`` (Brownian-like); otherwise the
  auxiliary linear drift ``β(t) + B(t)·x`` makes ``Φ_inv`` non-analytic
  and ``(H, F, c)`` are integrated via :func:`diffrax.diffeqsolve`
  (requires the ``[prebuilt-bffg]`` extra).

References
----------
- van der Meulen, F. H. & Sommer, S. (2025). *Backward Filtering
  Forward Guiding.* Journal of Machine Learning Research 26(281), 1–51.
  https://arxiv.org/abs/2505.18239
- van der Meulen, F. H. & Schauer, M. (2020-2022). *Automatic Backward
  Filtering Forward Guiding for Markov processes and graphical models.*
  https://arxiv.org/abs/2010.03509
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky
from jax.scipy.stats import multivariate_normal

from ..core.sweep import SweepFn, down, up
from ..core.tree import Tree
from .sde import dot, dts, forward, solve


# ── shared leaf-message helper ──────────────────────────────────────
def _canonical_leaf_messages(
    leaf_values: jax.Array,
    obs_var: float | jax.Array,
    *,
    n: int,
    d: int = 1,
) -> dict[str, jax.Array]:
    """Per-leaf ``(H_T, F_T, c_T)`` seeds for a BFFG up-sweep with iid
    Gaussian observations ``y_i ~ N(x_i, obs_var · I_n)``."""
    n_leaves = leaf_values.shape[0]
    H_T_leaf = jnp.eye(n) / obs_var
    H_T = jnp.broadcast_to(H_T_leaf, (n_leaves, n, n))
    F_T = jax.vmap(lambda v: (H_T_leaf @ v.reshape((n, d))).flatten())(leaf_values)

    Sigma_leaf = obs_var * jnp.eye(n)
    c_T = jax.vmap(
        lambda v: jax.vmap(
            lambda vc: multivariate_normal.logpdf(jnp.zeros(n), vc, Sigma_leaf)
        )(v.reshape((n, d)).T)
    )(leaf_values)

    return {"H_T": H_T, "F_T": F_T, "c_T": c_T}


# ── leaf initialization helpers ─────────────────────────────────────
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
    msgs = _canonical_leaf_messages(leaf_values, obs_var, n=n, d=d)
    return tree.at[tree.topology.is_leaf].set(**msgs)


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

    msgs = _canonical_leaf_messages(leaf_values, obs_var, n=n, d=d)
    tree = tree.at[leaf_mask].set(**msgs, v_T=leaf_values)

    if root_value is not None:
        root_value = jnp.asarray(root_value)
        if "v_0" in tree.schema:
            tree = tree.set(
                v_0=jnp.broadcast_to(root_value, (tree.size, n * d))
            )
        if "v_T" in tree.schema:
            # Seed v_T at inner nodes (leaves keep their observed values).
            tree = tree.at[~leaf_mask].set(
                v_T=jnp.broadcast_to(root_value, (tree.size - n_leaves, n * d)),
            )

    return tree


# ── Gaussian BFFG: up sweep (backward filter) ───────────────────────
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

        def per_child(edge_length, c_T_child, F_T_child, H_T_child):
            v_T = solve(H_T_child, F_T_child)
            covar = edge_length * a(v_T, params)
            invPhi_0 = jnp.eye(n) + H_T_child @ covar
            H_0 = jnp.linalg.solve(invPhi_0, H_T_child)
            F_0 = solve(invPhi_0, F_T_child)
            # c_0 = c_T + 0.5 v_T^T (H_T - H_0) v_T - 0.5 log|invPhi_0|,
            # per d-column. See van der Meulen & Sommer (2025) §3.
            log_det_phi_inv = jnp.linalg.slogdet(invPhi_0)[1]
            c_0 = jax.vmap(
                lambda v_T_col, c_T_col: c_T_col
                + 0.5 * v_T_col @ (H_T_child - H_0) @ v_T_col
                - 0.5 * log_det_phi_inv,
                in_axes=(1, 0),
            )(v_T.reshape((n, d)), c_T_child)
            return c_0, F_0, H_0

        c_0s, F_0s, H_0s = jax.vmap(per_child)(
            children.edge_length, children.c_T, children.F_T, children.H_T
        )
        return {"c_T": c_0s.sum(0), "F_T": F_0s.sum(0), "H_T": H_0s.sum(0)}

    return _sweep


# ── Gaussian BFFG: unconditional forward sampling ───────────────────
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


# ── Gaussian BFFG: conditional forward sampling (forward-guided) ────
def gaussian_down_conditional(n: int, a, d: int = 1) -> SweepFn:
    """Conditional forward sampling for nonlinear Gaussian BFFG.

    Implements the BFFG-guided proposal of van der Meulen & Sommer (2025),
    Theorem 14. For each non-root node, draws a sample of its state given
    its parent's current state (just sampled by the previous level of the
    down sweep) and the subtree canonical message ``(F_T, H_T)`` at this
    node (set by :func:`gaussian_up`). The per-edge log importance-weight
    correction is written to ``logw``.

    **Model assumed by this prebuilt.** The true Markov transition along
    each edge is

        X_t | X_s = x  ~  N(μ(x), Q(x)),     Q(x) = ℓ · a(x, params),

    with ``μ(x) = x`` (identity drift, i.e. Brownian-style increment) and
    ``Q(x)`` possibly state-dependent. The BFFG auxiliary used for
    backward filtering is the linearisation

        X̃_t | X̃_s = x  ~  N(Φ x + β, Q̃),    Φ = I,  β = 0,  Q̃ = ℓ · a(v_T, params),

    where ``v_T = H_T⁻¹ F_T`` is the canonical posterior mean from the
    child's message — matching the linearisation point used by
    :func:`gaussian_up`.

    **Sampling (Theorem 14, step 2).** The guided proposal at the child is

        X°_t | X°_s = x  ~  N^can(F + Q(x)⁻¹ x,  H + Q(x)⁻¹).

    **Importance weight (Theorem 14, step 3).**

        w(x) = (P g)(x) / (P̃ g)(x)
             = φ(H⁻¹F;  μ(x),  Q(x) + H⁻¹) / φ(H⁻¹F;  Φx + β,  Q̃ + H⁻¹).

    In the pure linear-Gaussian limit (``a`` independent of ``v``) we have
    ``Q(x) = Q̃`` and ``μ(x) = Φ x + β``, so the two densities coincide
    and ``logw ≡ 0`` exactly — matching the paper's remark on p.16 that
    ``w ≡ 1`` whenever the true dynamics are linear. State-dependent ``a``
    drives ``logw`` away from zero and produces the importance correction
    that BFFG-MCMC then uses.

    Args:
        n: latent state dimension.
        a: ``(v, params) -> (n, n)`` diffusion covariance per unit edge
            length. Evaluated at the parent's value for the true ``Q(x)``
            and at the child's canonical mean ``v_T`` for the auxiliary
            ``Q̃`` — matching :func:`gaussian_up`'s linearisation.
        d: data dimension. The canonical message factors across the ``d``
            iid data slices.
    """

    @down(
        reads=("noise", "edge_length", "F_T", "H_T"),
        reads_parent=("value",),
        writes=("value", "logw"),
    )
    def _sweep(node, parent, params):
        H_T_c = node.H_T
        F_T_c = node.F_T
        var = node.edge_length

        # Linearization point matching gaussian_up.
        v_T = solve(H_T_c, F_T_c)                            # (n·d,)

        # ── Conditional Gaussian sample (Theorem 14, step 2). With μ(x)=x,
        # the canonical posterior given the parent is
        #     N^can(F + Q(x)⁻¹·x,  H + Q(x)⁻¹).
        x = parent.value
        Q_true = var * a(x, params)                          # (n, n) — Q(x)
        invH = jnp.linalg.solve(jnp.eye(n) + Q_true @ H_T_c, Q_true)  # (n, n)
        mu = dot(invH, F_T_c + solve(Q_true, x))
        new_value = mu + dot(
            cholesky(invH, lower=True, check_finite=False), node.noise
        )

        # ── Importance weight (Theorem 14, step 3). With μ(x)=x, Φ=I, β=0
        # the two means coincide and logw reduces to the log-ratio of two
        # Gaussians at H⁻¹F that differ only in their covariance term
        # (Q(x) vs Q̃ = Q(v_T)). Linear case ⇒ both covariances equal ⇒
        # logw = 0 identically.
        Q_aux = var * a(v_T, params)                         # (n, n) — Q̃
        H_inv = jnp.linalg.inv(H_T_c)                        # H⁻¹ (n, n)
        C_true = Q_true + H_inv
        C_aux  = Q_aux  + H_inv
        logw = jnp.sum(jax.vmap(
            lambda v_T_col, x_col:
                multivariate_normal.logpdf(v_T_col, x_col, C_true)
                - multivariate_normal.logpdf(v_T_col, x_col, C_aux),
            in_axes=(1, 1),
        )(v_T.reshape((n, d)), x.reshape((n, d))))

        return {"value": new_value, "logw": logw}

    return _sweep


# ── SDE BFFG: backward filter over one edge ────────────────────────
def backward_filter(
    dts: jax.Array,
    params,
    c_T: jax.Array,
    v_T: jax.Array,
    F_T: jax.Array,
    H_T: jax.Array,
    tildea0: jax.Array,
    tildeaT: jax.Array,
    *,
    B=None,
    beta=None,
):
    """Backward filter over one SDE edge.

    Propagates ``(c, F, H)`` from the edge's lower end (child) to its
    upper end (parent). Closed form when ``B = β = None``; otherwise
    integrates ``(H, F, c)`` via diffrax and additionally returns the
    per-step ``F_t, H_t`` series needed by :func:`forward_guided`.

    Args:
        dts: time discretization, shape ``(n_steps,)``. Total edge time is
            ``T = dts.sum()``.
        params: passed through to ``B``, ``beta``.
        c_T, v_T, F_T, H_T: canonical Gaussian parameters at the lower end.
        tildea0, tildeaT: ``(n, n)`` auxiliary diffusivity at ``t=0`` and
            ``t=T``; ``tildea(t)`` is linearly interpolated.
        B: optional ``(t, params) -> (n, n)`` linear drift matrix.
        beta: optional ``(t, params) -> (n,)`` affine drift offset.
    """
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
    T = dts.sum()
    # ∫_0^T tildea(u) du = (T/2)·(tildeaT + tildea0) under linear interpolation.
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
    the edge's lower end to its upper end) via
    :func:`diffrax.diffeqsolve` with an adaptive PI step-size controller
    on the Tsit5 (5th-order Tsitouras) solver. Returns ``H_t``, ``F_t``
    indexed by *step number* (``0..n_steps-1``), with ``H_t[i]`` /
    ``F_t[i]`` at the END of step ``i`` — matching :func:`forward_guided`'s
    expectations.

    diffrax is an optional dependency (extra ``[prebuilt-bffg]``); the
    closed-form path doesn't need it.
    """
    try:
        import diffrax
    except ImportError as e:
        raise ImportError(
            "ODE-integrated backward_filter requires diffrax. Install via "
            "`uv sync --extra prebuilt-bffg` or `pip install 'hyperiax[prebuilt-bffg]'`."
        ) from e

    n = tildeaT.shape[0]
    d = v_T.size // n
    n_steps = dts.shape[0]
    T = dts.sum()

    nn = n * n
    nd = n * d

    def tildea(t):
        return tildeaT * (t / T) + tildea0 * (1 - t / T)

    def vector_field(tau, y, args):
        # Recover (H, F, c) from the packed state.
        Ht = y[:nn].reshape((n, n))
        Ft_flat = y[nn:nn + nd]
        Ft_mat = Ft_flat.reshape((n, d))

        t = T - tau
        Bt = B(t, args) if B is not None else jnp.zeros((n, n))
        betat = beta(t, args) if beta is not None else jnp.zeros(n)
        ta = tildea(t)

        # Forward-in-t derivatives (van der Meulen et al. eqns).
        dH = -Bt.T @ Ht - Ht @ Bt + Ht @ ta @ Ht
        dF_mat = -Bt.T @ Ft_mat + Ht @ ta @ Ft_mat + (Ht @ betat)[:, None]
        dc = -(
            betat @ Ft_mat
            + 0.5 * jnp.einsum("id,ij,jd->d", Ft_mat, ta, Ft_mat)
            - 0.5 * jnp.trace(Ht @ ta) * jnp.ones(d)
        )
        # diffrax integrates dy/dτ; we want τ = T - t so dy/dτ = -dy/dt.
        return -jnp.concatenate([dH.flatten(), dF_mat.flatten(), dc])

    y0 = jnp.concatenate([H_T.flatten(), F_T, c_T])
    # Save points: uniform grid spanning [0, T] in τ-time (n_steps+1 points).
    ts_tau = jnp.linspace(0.0, T, n_steps + 1)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=0.0,
        t1=T,
        dt0=T / n_steps,
        y0=y0,
        args=params,
        saveat=diffrax.SaveAt(ts=ts_tau),
        stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-9),
    )
    solution = sol.ys  # (n_steps+1, packed_dim), indexed by τ from 0 to T
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


# ── SDE BFFG: forward-guided bridge ────────────────────────────────
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
    tildea0: jax.Array,
    tildeaT: jax.Array,
    B=None,
    beta=None,
):
    """Forward-guided SDE bridge.

    Integrates ``dx = b(t,x) dt + a(x)·(F(t) − H(t)·x) dt + σ(x) dW`` from
    ``x0``. Closed form needs ``F_T, H_T``; the ODE path needs the
    per-step ``F_t, H_t`` series from :func:`backward_filter`. Returns
    ``(Xs, logpsi)``.
    """
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


# ── SDE BFFG: up sweep (backward filter on each edge) ──────────────
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


# ── SDE BFFG: unconditional forward sampling ───────────────────────
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


# ── SDE BFFG: conditional forward sampling (forward-guided bridge) ──
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

        guide_kwargs: dict = {"tildea0": tildea0, "tildeaT": tildeaT}
        if B is None and beta is None:
            guide_kwargs["F_T"] = node.F_T
            guide_kwargs["H_T"] = node.H_T
        else:
            # Re-derive F_t, H_t for this edge via the ODE filter.
            filt = backward_filter(
                _dts, params, node.c_T, node.v_T, node.F_T, node.H_T,
                tildea0=tildea0, tildeaT=tildeaT, B=B, beta=beta,
            )
            guide_kwargs.update(
                F_t=filt["F_t"], H_t=filt["H_t"], B=B, beta=beta,
            )
        Xs, logpsi = forward_guided(
            x0, _dts, _dWs, b, sigma, params, a, **guide_kwargs,
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
