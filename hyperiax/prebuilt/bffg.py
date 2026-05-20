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
  (requires the ``[prebuilt]`` extra).

Naming convention
-----------------
The BFFG auxiliary process (``X̃``) shows up everywhere; we keep its
names visually distinct from the true process by appending ``_aux``:

==================  =====================================  =========================
Symbol              Name                                   Meaning
==================  =====================================  =========================
``ã(t)``            ``a_aux`` (callable)                    Auxiliary diffusivity at t
``ã(0)``, ``ã(T)``  ``a_aux_0``, ``a_aux_T``                Endpoint values of ``ã``
``b̃(t, x)``         ``b_aux``                               Auxiliary drift (B·x + β)
``r̃(t)``            ``r_aux``                               Score ``F(t) − H(t)·x``
==================  =====================================  =========================

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
from jax.scipy.linalg import cho_solve, cholesky, solve_triangular

from ..core.sweep import SweepFn, down, up
from ..core.tree import Tree
from .sde import dot, dts, forward, solve


# ── schema helpers ─────────────────────────────────────────────────
def bffg_gaussian_schema(n: int, d: int = 1) -> dict[str, tuple[int, ...]]:
    """Tree schema for Gaussian BFFG (up + conditional down).

    Bundles every field :func:`gaussian_up`, :func:`gaussian_down_unconditional`,
    and :func:`gaussian_down_conditional` read or write. Pair with
    :func:`init_gaussian_leaves` to seed the canonical messages.

    Args:
        n: latent state dimension.
        d: data dimension (1 for scalar, 2/3 for landmarks).
    """
    return {
        "edge_length": (),
        "value": (n * d,),
        "noise": (n * d,),
        "c_T": (d,),
        "F_T": (n * d,),
        "H_T": (n, n),
        "logw": (),
    }


def bffg_sde_schema(
    n: int,
    d: int = 1,
    n_steps: int = 100,
) -> dict[str, tuple[int, ...]]:
    """Tree schema for SDE BFFG (up + propagate-linearization + down).

    Bundles every field :func:`sde_up`, :func:`sde_down_unconditional`,
    :func:`sde_down_conditional`, and :func:`propagate_linearization`
    read or write. Pair with :func:`init_sde_tree` to seed the canonical
    messages plus the linearization points ``v_0`` / ``v_T``.

    Args:
        n: latent state dimension.
        d: data dimension.
        n_steps: time discretization for each edge.
    """
    return {
        "edge_length": (),
        "value": (n_steps + 1, n * d),
        "noise": (n_steps, n * d),
        "c_T": (d,),
        "F_T": (n * d,),
        "H_T": (n, n),
        "F_t": (n_steps, n * d),
        "H_t": (n_steps, n, n),
        "v_T": (n * d,),
        "v_0": (n * d,),
        "logpsi": (),
    }


# ── shared leaf-message helper ──────────────────────────────────────
def _canonical_leaf_messages(
    leaf_values: jax.Array,
    obs_var: float | jax.Array,
    *,
    n: int,
    d: int = 1,
) -> dict[str, jax.Array]:
    """Per-leaf ``(H_T, F_T, c_T)`` seeds for a BFFG up-sweep with iid
    Gaussian observations ``y_i ~ N(x_i, obs_var · I_n)``.

    Closed-form: ``H_T = I_n / σ²`` is constant across leaves; ``F_T = y / σ²``
    since ``H_T_leaf`` is a scaled identity; ``c_T`` is the log-density of
    ``N(0; y_col, σ²·I_n)`` per d-column, computed analytically (no
    multivariate_normal call).
    """
    n_leaves = leaf_values.shape[0]
    H_T_leaf = jnp.eye(n) / obs_var
    H_T = jnp.broadcast_to(H_T_leaf, (n_leaves, n, n))
    F_T = leaf_values / obs_var

    sq = jnp.sum(leaf_values.reshape((n_leaves, n, d)) ** 2, axis=1)  # (n_leaves, d)
    log_norm = -0.5 * n * (jnp.log(2.0 * jnp.pi) + jnp.log(obs_var))
    c_T = log_norm - 0.5 * sq / obs_var

    return {"H_T": H_T, "F_T": F_T, "c_T": c_T}


# ── leaf initialization helpers ─────────────────────────────────────
def init_gaussian_leaves(
    tree: Tree,
    leaf_values: jax.Array,
    obs_var: float | jax.Array,
    *,
    n: int,
    d: int = 1,
    root_value: jax.Array | None = None,
) -> Tree:
    """Seed ``H_T``, ``F_T``, ``c_T`` at the leaves for a Gaussian up-sweep.

    Args:
        tree: a Tree with at least the fields ``H_T (n, n)``, ``F_T (n·d,)``,
            and ``c_T (d,)`` declared in its schema — typically built with
            :func:`bffg_gaussian_schema`.
        leaf_values: array of shape ``(n_leaves, n·d)`` carrying the
            observed leaf states.
        obs_var: scalar observation noise variance ``τ²``. Per-leaf
            precision is ``I_n / τ²``.
        n: latent state dimension.
        d: data dimension.
        root_value: optional ``(n·d,)`` root state. If given, every node's
            ``value`` field is broadcast to this — useful when the
            conditional down sweep needs a fixed starting point.

    Returns:
        A new Tree with the leaf rows of ``H_T``, ``F_T``, ``c_T`` set
        (and optionally ``value`` everywhere).
    """
    msgs = _canonical_leaf_messages(leaf_values, obs_var, n=n, d=d)
    tree = tree.at[tree.topology.is_leaf].set(**msgs)
    if root_value is not None:
        root_value = jnp.asarray(root_value)
        if "value" in tree.schema:
            tree = tree.set(value=jnp.broadcast_to(root_value, (tree.size, n * d)))
    return tree


def init_sde_tree(
    tree: Tree,
    leaf_values: jax.Array,
    obs_var: float | jax.Array,
    *,
    n: int,
    d: int = 1,
    root_value: jax.Array | None = None,
) -> Tree:
    """Seed leaf canonical messages and (optionally) ``v_0`` / ``v_T``
    everywhere from a single root linearization point.

    Despite the name, this writes more than just the leaves: when
    ``root_value`` is given, ``v_0`` and ``v_T`` are broadcast to every
    node (the inner-node ``v_T`` will be overwritten by :func:`sde_up`).

    The user is responsible for declaring the matching fields on the
    tree's schema — :func:`bffg_sde_schema` bundles them all.
    """
    leaf_mask = tree.topology.is_leaf
    n_leaves = int(leaf_mask.sum())

    msgs = _canonical_leaf_messages(leaf_values, obs_var, n=n, d=d)
    tree = tree.at[leaf_mask].set(**msgs, v_T=leaf_values)

    if root_value is not None:
        root_value = jnp.asarray(root_value)
        if "v_0" in tree.schema:
            tree = tree.set(v_0=jnp.broadcast_to(root_value, (tree.size, n * d)))
        if "v_T" in tree.schema:
            # Seed v_T at inner nodes (leaves keep their observed values).
            tree = tree.at[~leaf_mask].set(
                v_T=jnp.broadcast_to(root_value, (tree.size - n_leaves, n * d)),
            )

    return tree


# ── Gaussian BFFG: up sweep (backward filter) ───────────────────────
def _canonical_pullback_step(H_T_child, F_T_child, c_T_child, covar, n, d):
    """One edge of the canonical (H, F, c) pullback under the Gaussian
    transition with covariance ``covar`` and identity drift.

    Shared by :func:`gaussian_up` and :func:`_backward_filter_closed_form`.
    Both implement the formula

        H_0 = (I + H_T·Q)⁻¹ · H_T
        F_0 = (I + H_T·Q)⁻¹ · F_T
        c_0 = c_T + ½ v_T^T (H_T − H_0) v_T − ½ log|I + H_T·Q|

    where ``Q = covar`` is the edge covariance for the closed-form
    Gaussian case and ``(T/2)(ã_T + ã_0)`` for the linear-tildea
    closed-form SDE case. The c_0 update is per d-column.
    """
    phi_aux_inv = jnp.eye(n) + H_T_child @ covar
    H_0 = jnp.linalg.solve(phi_aux_inv, H_T_child)
    F_0 = solve(phi_aux_inv, F_T_child)
    v_T = solve(H_T_child, F_T_child)
    log_det = jnp.linalg.slogdet(phi_aux_inv)[1]
    c_0 = jax.vmap(
        lambda v_col, c_col: c_col + 0.5 * v_col @ (H_T_child - H_0) @ v_col - 0.5 * log_det,
        in_axes=(1, 0),
    )(v_T.reshape((n, d)), c_T_child)
    return c_0, F_0, H_0


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
            return _canonical_pullback_step(H_T_child, F_T_child, c_T_child, covar, n, d)

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
        H_T_child = node.H_T
        F_T_child = node.F_T
        var = node.edge_length

        # Linearization point matching gaussian_up.
        v_T = solve(H_T_child, F_T_child)  # (n·d,)

        # ── Conditional Gaussian sample (Theorem 14, step 2). With μ(x)=x,
        # the canonical posterior given the parent is
        #     N^can(F + Q(x)⁻¹·x,  H + Q(x)⁻¹).
        x = parent.value
        Q_true = var * a(x, params)  # (n, n) — Q(x)
        inv_H = jnp.linalg.solve(jnp.eye(n) + Q_true @ H_T_child, Q_true)  # (n, n)
        mu = dot(inv_H, F_T_child + solve(Q_true, x))
        new_value = mu + dot(cholesky(inv_H, lower=True, check_finite=False), node.noise)

        # ── Importance weight (Theorem 14, step 3). With μ(x)=x, Φ=I, β=0
        # the two means and the d-iid residuals coincide; only the
        # covariances differ (Q(x) vs Q̃ = Q(v_T)). Factor each covariance
        # once total via Cholesky and accumulate the quadratic forms over
        # all d columns — avoids one fresh Cholesky per d-column inside
        # multivariate_normal.logpdf. Linear case ⇒ Q == Q̃ ⇒ logw = 0.
        Q_aux = var * a(v_T, params)  # (n, n) — Q̃
        L_H = cholesky(H_T_child, lower=True, check_finite=False)
        H_inv = cho_solve((L_H, True), jnp.eye(n))
        C_true = Q_true + H_inv
        C_aux = Q_aux + H_inv
        L_true = cholesky(C_true, lower=True, check_finite=False)
        L_aux = cholesky(C_aux, lower=True, check_finite=False)
        diff = (v_T - x).reshape((n, d))
        y_true = solve_triangular(L_true, diff, lower=True)
        y_aux = solve_triangular(L_aux, diff, lower=True)
        logdet_true = 2.0 * jnp.sum(jnp.log(jnp.diag(L_true)))
        logdet_aux = 2.0 * jnp.sum(jnp.log(jnp.diag(L_aux)))
        logw = -0.5 * d * (logdet_true - logdet_aux) - 0.5 * (
            jnp.sum(y_true**2) - jnp.sum(y_aux**2)
        )

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
    a_aux_0: jax.Array,
    a_aux_T: jax.Array,
    *,
    B=None,
    beta=None,
):
    """Backward filter over one SDE edge.

    Propagates ``(c, F, H)`` from the edge's lower end (child) to its
    upper end (parent). Closed form when ``B = β = None`` (uses an
    analytical ``Φ̃⁻¹(t)``); otherwise integrates ``(H, F, c)`` via
    diffrax with an adaptive Tsit5+PID controller.

    Both paths return the same dict:
    ``{c_0, F_0, H_0, F_t, H_t}``. ``F_t`` / ``H_t`` are the per-step
    canonical-message series along the edge, indexed by step end
    (``i = 0`` → ``t = dts[0]``, ..., ``i = n_steps-1`` → ``t = T``).
    The matching :func:`forward_guided` consumes them directly so no
    per-step backward integration happens inside the bridge sampler.

    Args:
        dts: time discretization, shape ``(n_steps,)``. Total edge time is
            ``T = dts.sum()``.
        params: passed through to ``B``, ``beta``.
        c_T, v_T, F_T, H_T: canonical Gaussian parameters at the lower end.
        a_aux_0, a_aux_T: ``(n, n)`` auxiliary diffusivity at ``t=0`` and
            ``t=T``; ``a_aux(t)`` is linearly interpolated.
        B: optional ``(t, params) -> (n, n)`` linear drift matrix.
        beta: optional ``(t, params) -> (n,)`` affine drift offset.
    """
    if B is None and beta is None:
        return _backward_filter_closed_form(dts, c_T, v_T, F_T, H_T, a_aux_0, a_aux_T)
    return _backward_filter_ode(dts, params, c_T, v_T, F_T, H_T, a_aux_0, a_aux_T, B, beta)


def _backward_filter_closed_form(dts, c_T, v_T, F_T, H_T, a_aux_0, a_aux_T):
    n = a_aux_T.shape[0]
    d = v_T.size // n
    T = dts.sum()

    # Φ̃⁻¹(t) = I + H_T · (c1(t)·a_aux_T + c2(t)·a_aux_0)
    #        = I + c1(t) · H_T_aT + c2(t) · H_T_a0
    # where H_T_aT, H_T_a0 are loop-invariant. Hoist them once so the vmap'd
    # ``per_t`` below replaces 500 (n×n)@(n×n) matmuls with 500 elementwise
    # lerps; same identity also folds the t=0 endpoint into one matmul-free
    # combination (under linear a_aux, ∫_0^T a_aux = (T/2)(a_aux_T + a_aux_0)).
    H_T_aT = H_T @ a_aux_T
    H_T_a0 = H_T @ a_aux_0

    # t = 0 endpoint: integrated ã from 0 to T equals (T/2)·(ã_T + ã_0), so
    # share the canonical-pullback formula with gaussian_up by passing this
    # effective covariance.
    covar_0 = (T / 2.0) * (a_aux_T + a_aux_0)
    c_0, F_0, H_0 = _canonical_pullback_step(H_T, F_T, c_T, covar_0, n, d)

    # Per-step F_t, H_t via the analytical Φ̃⁻¹(t). Batched via vmap so XLA
    # fuses the n_steps solves — matches the ODE path's output schema.
    ts = jnp.cumsum(dts)

    def per_t(t):
        c1 = -(t**2 - T**2) / (2.0 * T)
        c2 = ((T - t) ** 2) / (2.0 * T)
        phi_aux_inv_t = jnp.eye(n) + c1 * H_T_aT + c2 * H_T_a0
        H_t_i = jnp.linalg.solve(phi_aux_inv_t, H_T)
        F_t_i = solve(phi_aux_inv_t, F_T)
        return H_t_i, F_t_i

    H_t, F_t = jax.vmap(per_t)(ts)

    return {"c_0": c_0, "F_0": F_0, "H_0": H_0, "F_t": F_t, "H_t": H_t}


def _backward_filter_ode(dts, params, c_T, v_T, F_T, H_T, a_aux_0, a_aux_T, B, beta):
    """ODE-integrated path for general linear ``B(t)``, ``β(t)``.

    Integrates ``(H, F, c)`` with τ = T - t (so τ ∈ [0, T] runs from
    the edge's lower end to its upper end) via
    :func:`diffrax.diffeqsolve` with an adaptive PI step-size controller
    on the Tsit5 (5th-order Tsitouras) solver. Returns ``H_t``, ``F_t``
    indexed by *step number* (``0..n_steps-1``), with ``H_t[i]`` /
    ``F_t[i]`` at the END of step ``i`` — matching :func:`forward_guided`'s
    expectations.

    diffrax is an optional dependency (extra ``[prebuilt]``); the
    closed-form path doesn't need it.
    """
    try:
        import diffrax
    except ImportError as e:
        raise ImportError(
            "ODE-integrated backward_filter requires diffrax. Install via "
            "`uv sync --extra prebuilt` or `pip install 'hyperiax[prebuilt]'`."
        ) from e

    n = a_aux_T.shape[0]
    d = v_T.size // n
    n_steps = dts.shape[0]
    T = dts.sum()

    nn = n * n
    nd = n * d

    def a_aux(t):
        return a_aux_T * (t / T) + a_aux_0 * (1 - t / T)

    def vector_field(tau, y, args):
        # Recover (H, F, c) from the packed state.
        Ht = y[:nn].reshape((n, n))
        Ft_flat = y[nn : nn + nd]
        Ft_mat = Ft_flat.reshape((n, d))

        t = T - tau
        Bt = B(t, args) if B is not None else jnp.zeros((n, n))
        betat = beta(t, args) if beta is not None else jnp.zeros(n)
        at = a_aux(t)

        # Forward-in-t derivatives (van der Meulen et al. eqns).
        dH = -Bt.T @ Ht - Ht @ Bt + Ht @ at @ Ht
        dF_mat = -Bt.T @ Ft_mat + Ht @ at @ Ft_mat + (Ht @ betat)[:, None]
        dc = -(
            betat @ Ft_mat
            + 0.5 * jnp.einsum("id,ij,jd->d", Ft_mat, at, Ft_mat)
            - 0.5 * jnp.trace(Ht @ at) * jnp.ones(d)
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
    F_0 = state_t0[nn : nn + nd]
    c_0 = state_t0[nn + nd :]

    H_t = per_step_end[:, :nn].reshape((n_steps, n, n))
    F_t = per_step_end[:, nn : nn + nd]

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
    F_t: jax.Array,
    H_t: jax.Array,
    a_aux_0: jax.Array,
    a_aux_T: jax.Array,
    B=None,
    beta=None,
):
    """Forward-guided SDE bridge.

    Integrates ``dx = b(t,x) dt + a(x)·(F(t) − H(t)·x) dt + σ(x) dW`` from
    ``x0``. ``F_t``/``H_t`` are the per-step canonical-message arrays
    produced by :func:`backward_filter` (either path returns them) and
    cached on each child node by :func:`sde_up`. ``B``/``β`` control the
    auxiliary drift used by the importance-weight ``logpsi``; pass
    ``None`` for the Brownian-style auxiliary.

    Returns ``(Xs, logpsi)`` where ``Xs`` has shape ``(n_steps + 1, n·d)``
    and ``logpsi`` is a scalar.
    """
    n = a_aux_T.shape[0]
    d = x0.size // n
    ts = jnp.cumsum(dts)
    T = ts[-1]

    def a_aux(t):
        return a_aux_T * (t / T) + a_aux_0 * (1 - t / T)

    # b_aux ≡ 0 when neither B nor β is supplied → drift_diff = b in the body;
    # pass None as a sentinel so the body can skip the no-op subtraction.
    if B is None and beta is None:
        b_aux = None
    else:

        def b_aux(t, X, params_):
            betat = beta(t, params_) if beta is not None else jnp.zeros(n)
            if B is None:
                return jnp.broadcast_to(betat[:, None], (n, d)).flatten()
            Bt = B(t, params_)
            return (Bt @ X.reshape((n, d)) + betat[:, None]).flatten()

    def step(carry, val):
        i, X, logpsi = carry
        dt, dW = val
        t = ts[i]
        return _bridge_step_body(
            i,
            X,
            logpsi,
            dt,
            dW,
            t,
            H_t[i],
            F_t[i],
            b,
            sigma,
            a,
            params,
            a_aux(t),
            b_aux=b_aux,
            n=n,
            d=d,
        )

    (_, X, logpsi), Xs = jax.lax.scan(step, (0, x0, 0.0), (dts, dWs))
    return jnp.vstack((Xs, X)), logpsi


def _bridge_step_body(
    i,
    X,
    logpsi,
    dt,
    dW,
    t,
    H_t_i,
    F_t_i,
    b,
    sigma,
    a,
    params,
    a_aux_t,
    *,
    b_aux,
    n,
    d,
):
    """Single Euler step of the guided SDE; shared between closed-form and ODE paths.

    ``b_aux`` is a callable ``(t, X, params) -> (n·d,)`` or ``None``.
    ``None`` is the B=β=None case where the auxiliary drift is zero — the body
    then uses ``drift_diff = b`` directly instead of computing ``b - 0``.
    """
    r_aux = F_t_i - dot(H_t_i, X)
    if sigma is not None:
        _sigma = sigma(X, params)
        _a = jnp.einsum("ij,kj->ik", _sigma, _sigma)
    else:
        _a = a(X, params)
        _sigma = cholesky(_a, lower=True, check_finite=False)

    b_val = b(t, X, params)
    Xtp1 = X + b_val * dt + dot(_a, r_aux) * dt + dot(_sigma, dW)

    a_minus_aux = _a - a_aux_t
    drift_diff = b_val if b_aux is None else b_val - b_aux(t, X, params)
    logpsi_tp1 = (
        logpsi
        + (
            jnp.dot(drift_diff, r_aux)
            - 0.5 * d * jnp.einsum("ij,ji->", a_minus_aux, H_t_i)
            + 0.5
            * jnp.einsum(
                "ij,jd,id->",
                a_minus_aux,
                r_aux.reshape((n, d)),
                r_aux.reshape((n, d)),
            )
        )
        * dt
    )
    return (i + 1, Xtp1, logpsi_tp1), X


# ── SDE BFFG: up sweep (backward filter on each edge) ──────────────
def sde_up(n_steps: int, a, *, B=None, beta=None) -> SweepFn:
    """SDE BFFG up-sweep.

    Each parent's posterior canonical params ``(c_T, F_T, H_T)`` and
    posterior mean ``v_T`` are computed by:

    1. Per child, running :func:`backward_filter` over the edge to
       propagate the child's ``(c_T, F_T, H_T)`` upward to
       ``(c_0, F_0, H_0)`` and (as a byproduct) the per-step
       ``F_t, H_t`` series along the edge.
    2. Summing the per-child messages into the parent.
    3. Solving ``v_T = H_T⁻¹ F_T``.

    Each child node receives the ``F_t``/``H_t`` arrays via
    ``writes_children`` — :func:`sde_down_conditional` reads them
    directly, no per-edge backward integration during sampling.

    Schema requirement: see :func:`bffg_sde_schema`.

    Args:
        n_steps: time discretization for the per-edge filter (sets the
            length of the ``F_t``/``H_t`` series).
        a: per-edge diffusion covariance ``(v, params) -> (n, n)``.
        B: optional ``(t, params) -> (n, n)``. If ``B`` or ``β`` is
            provided the per-edge filter switches to the ODE path.
        beta: optional ``(t, params) -> (n,)``.
    """

    @up(
        reads_children=("edge_length", "v_0", "c_T", "v_T", "F_T", "H_T"),
        writes=("c_T", "v_T", "F_T", "H_T"),
        writes_children=("F_t", "H_t"),
    )
    def _sweep(node, children, params):
        def per_child(edge_length, v_0_c, c_T_c, v_T_c, F_T_c, H_T_c):
            _dts = dts(T=edge_length, n_steps=n_steps)
            return backward_filter(
                _dts,
                params,
                c_T_c,
                v_T_c,
                F_T_c,
                H_T_c,
                a_aux_0=a(v_0_c, params),
                a_aux_T=a(v_T_c, params),
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
            "F_t": msgs["F_t"],
            "H_t": msgs["H_t"],
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
    n_steps: int,
    b,
    sigma,
    a,
    *,
    B=None,
    beta=None,
) -> SweepFn:
    """Conditional forward sampling via :func:`forward_guided`.

    Reads the per-edge ``F_t``/``H_t`` cached on each child node by
    :func:`sde_up` (no per-edge backward integration is repeated here).
    ``B``/``β`` only enter through the bridge's ``logpsi`` correction —
    pass them through if you use them in :func:`sde_up`.

    Schema requirement: see :func:`bffg_sde_schema`.
    """

    @down(
        reads=("noise", "edge_length", "v_0", "v_T", "F_t", "H_t"),
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
            F_t=node.F_t,
            H_t=node.H_t,
            a_aux_0=a(node.v_0, params),
            a_aux_T=a(node.v_T, params),
            B=B,
            beta=beta,
        )
        return {"value": Xs, "logpsi": logpsi}

    return _sweep


# ── linearization propagation (down sweep: v_0 of node ← v_T of parent) ──
@down(reads_parent=("v_T",), writes=("v_0",))
def propagate_linearization(node, parent, params):
    """Set each non-root node's ``v_0`` equal to its parent's ``v_T``.

    Run this after :func:`sde_up` so that the linearization point for
    ``a_aux`` on each edge reflects the posterior mean at the parent.
    The root's ``v_0`` should be initialized separately (typically to the
    root prior value).
    """
    return {"v_0": parent.v_T}
