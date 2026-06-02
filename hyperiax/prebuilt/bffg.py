"""Backward Filtering Forward Guiding (BFFG) on rooted directed trees.

Implements the framework of van der Meulen & Sommer (2025), *Backward Filtering
Forward Guiding*, JMLR 26(281), 1–51 (`arXiv:2505.18239
<https://arxiv.org/abs/2505.18239>`_). Each tree edge is either:

- **discrete** (§6.1, Theorem 14): a 1-step nonlinear-Gaussian Markov kernel
  :math:`X_t \\mid X_{pa(t)}=x \\sim \\mathcal N(\\mu(x), Q(x))`, approximated
  by a tractable linear-Gaussian auxiliary :math:`\\tilde X_t \\sim
  \\mathcal N(\\Phi x + \\beta, Q_{\\text{aux}})`; or
- **continuous** (§7.1, Theorem 23): an SDE
  :math:`dX_u = b(u, X_u)\\,du + \\sigma(u, X_u)\\,dW_u` with a linear
  auxiliary :math:`d\\tilde X_u = (B(u)\\tilde X_u + \\beta(u))\\,du +
  \\tilde\\sigma(u)\\,dW_u`.

The auxiliary is evaluated at a per-edge **linearisation anchor** (Algorithm 3
§7.1) that the user iteratively refines toward the BFFG posterior mean.

Typical workflow::

    bf = continuous_bf_sweep(n_steps, B, beta, sigma_tilde)
    refine = continuous_refine_anchor()
    fg = continuous_fg_sweep(n_steps, b, sigma, B, beta, sigma_tilde)

    tree = init_continuous_tree(empty, leaf_obs, obs_var=tau_sq, d=d,
                                n_steps=n_steps, root_val=x_root)
    for _ in range(n_lin_iters):
        tree = bf(tree, params=theta)
        tree = refine(tree, params=theta)
    tree = tree.set(zs=z.reshape((N, n_steps, d)))
    tree = fg(tree, params=theta)
    # tree.vals — guided bridge trajectories; tree.log_corr — per-edge
    # importance log-weights; tree.log_norm[0] — root marginal log-evidence.

Schemas: see :func:`discrete_schema` and :func:`continuous_schema`.

References:
    van der Meulen, F. H. & Sommer, S. (2025). Backward Filtering Forward
    Guiding. *JMLR* 26(281), 1–51. https://arxiv.org/abs/2505.18239
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, cholesky, solve_triangular

from ..core.sweep import SweepFn, down, up
from ..core.tree import Tree
from ..utils.ode import RK4, solve_ode
from ..utils.sde import EulerMaruyama, dot, solve, solve_sde


# ── schema helpers ─────────────────────────────────────────────────
def discrete_schema(d: int) -> dict[str, tuple[int, ...]]:
    """Field layout for a discrete-edge BFFG tree.

    Args:
        d: State dimension.

    Returns:
        Mapping of field name to per-node trailing shape, suitable for
        :class:`hyperiax.Schema.from_dict`. Fields:

        - ``val: (d,)`` — node state (filled by the forward sweep).
        - ``z: (d,)`` — i.i.d. standard-normal noise driving the forward sample.
        - ``ptnl: (d,)`` — canonical-form potential ``F`` at the node (BF).
        - ``prec: (d, d)`` — canonical-form precision ``H`` at the node (BF).
        - ``anchor: (d,)`` — per-edge linearisation point for the auxiliary
          kernel (Algorithm 3 §7.1); leaves are seeded at the observation,
          internal nodes refined via :func:`discrete_refine_anchor`.
        - ``log_norm: ()`` — canonical-message log-norm; sums up the tree so
          ``log_norm[root]`` is the marginal log-evidence.
        - ``log_corr: ()`` — per-edge importance log-weight from forward
          guiding (Theorem 14.3).
    """
    return {
        "val": (d,),
        "z": (d,),
        "ptnl": (d,),
        "prec": (d, d),
        "anchor": (d,),
        "log_norm": (),
        "log_corr": (),
    }


def continuous_schema(d: int, n_steps: int) -> dict[str, tuple[int, ...]]:
    """Field layout for a continuous-edge (SDE) BFFG tree.

    Args:
        d: State dimension.
        n_steps: Number of Euler-Maruyama / RK4 substeps per edge.

    Returns:
        Mapping of field name to per-node trailing shape. Each non-root node's
        edge is discretised into ``n_steps`` substeps (``n_steps + 1`` time
        points). Fields:

        - ``edge_len: ()`` — Δt for this node's incoming edge.
        - ``vals: (n_steps + 1, d)`` — forward-sampled trajectory along the edge.
        - ``zs: (n_steps, d)`` — i.i.d. standard-normal increments driving the SDE.
        - ``ptnls: (n_steps + 1, d)`` / ``precs: (n_steps + 1, d, d)`` —
          per-step canonical ``(F, H)`` trajectory along the edge (BF).
        - ``ptnl_v: (d,)`` / ``prec_v: (d, d)`` — vertex (fused) canonical
          message at this node; written by the up-sweep, used as the terminal
          condition for the next-level edge.
        - ``anchor: (d,)`` — child-end linearisation point of this node's edge
          (t = ``edge_len``); refined toward the BFFG posterior mean.
        - ``anchor_pa: (d,)`` — parent-end linearisation point of this node's
          edge (t = 0); cached on the child so the up-sweep can read both
          endpoints inside ``children.map(...)``.
        - ``log_norm: ()`` — canonical-message log-norm at the vertex; sums
          up the tree so ``log_norm[root]`` is the marginal log-evidence.
        - ``log_corr: ()`` — per-edge importance log-weight from forward
          guiding (Theorem 23 eq 32 / Remark 24).
    """
    return {
        "edge_len": (),
        "vals": (n_steps + 1, d),
        "zs": (n_steps, d),
        "ptnls": (n_steps + 1, d),
        "precs": (n_steps + 1, d, d),
        "ptnl_v": (d,),
        "prec_v": (d, d),
        "anchor": (d,),
        "anchor_pa": (d,),
        "log_norm": (),
        "log_corr": (),
    }


# ── shared leaf-message helper ──────────────────────────────────────
def _canonical_leaf_messages(
    leaf_obs: jax.Array,
    obs_var: float | jax.Array,
    *,
    d: int,
) -> dict[str, jax.Array]:
    """Canonical-form Gaussian leaf message.

    Returns ``(prec, ptnl, log_norm)`` for ``log p(y | x) = c + F·x - ½x'Hx``
    with ``H = I/obs_var``, ``F = y/obs_var``, and the constant
    ``c = -½d log(2π·obs_var) - ½ y'y / obs_var`` that turns the canonical
    message into an actual log-density at ``x``.
    """
    n_leaves = leaf_obs.shape[0]
    obs_var = jnp.asarray(obs_var, dtype=leaf_obs.dtype)
    prec = jnp.eye(d, dtype=leaf_obs.dtype) / obs_var
    prec = jnp.broadcast_to(prec, (n_leaves, d, d))
    ptnl = leaf_obs / obs_var
    log_norm = (
        -0.5 * d * jnp.log(2.0 * jnp.pi * obs_var) - 0.5 * jnp.sum(leaf_obs**2, axis=-1) / obs_var
    )  # (n_leaves,)
    return {"prec": prec, "ptnl": ptnl, "log_norm": log_norm}


# ── leaf initialization helpers ─────────────────────────────────────
def init_discrete_tree(
    tree: Tree,
    leaf_obs: jax.Array,
    obs_var: float | jax.Array,
    *,
    d: int,
    root_val: jax.Array | None = None,
    anchor_init: jax.Array | None = None,
) -> Tree:
    """Seed a discrete-edge BFFG tree with leaf observations and initial anchors.

    Writes leaves' canonical-form message ``(prec, ptnl, log_norm)`` from the
    isotropic-Gaussian likelihood :math:`y \\sim \\mathcal N(x, \\tau^2 I)` and
    seeds every node's ``anchor`` (leaves at their observation, others at
    ``anchor_init``). Optionally pins the root state.

    Args:
        tree: Empty tree with the schema returned by :func:`discrete_schema`.
        leaf_obs: ``(n_leaves, d)`` leaf observations in BFS leaf order.
        obs_var: Scalar observation variance :math:`\\tau^2`.
        d: State dimension.
        root_val: Optional ``(d,)`` value to pin at the root (writes the
            ``val`` field). Required by samplers that condition on a fixed
            root.
        anchor_init: Optional ``(d,)`` initial anchor for non-leaf nodes
            (defaults to ``root_val`` if given, else zeros).

    Returns:
        Tree with leaf canonical messages set, anchors seeded, and (optionally)
        the root pinned.
    """
    msgs = _canonical_leaf_messages(leaf_obs, obs_var, d=d)
    tree = tree.at[tree.topology.is_leaf].set(
        prec=msgs["prec"], ptnl=msgs["ptnl"], log_norm=msgs["log_norm"]
    )
    if root_val is not None:
        root_val = jnp.asarray(root_val)
        if "val" in tree.schema:
            tree = tree.at[tree.topology.is_root].set(val=root_val)
        else:
            raise ValueError("Schema must declare 'val' field to set root_val.")
    # Seed every node's anchor (per-edge linearisation point).
    # - Leaves: anchor = leaf_obs (the observed terminal state; this is the
    #   posterior mean at the leaf since prec_leaf=I/τ², ptnl_leaf=y/τ²).
    # - Inner/root: anchor = anchor_init (defaults to root_val or zeros).
    if "anchor" in tree.schema:
        if anchor_init is None:
            anchor_init = (
                jnp.zeros(d, dtype=leaf_obs.dtype) if root_val is None else jnp.asarray(root_val)
            )
        anchor_init = jnp.asarray(anchor_init)
        tree = tree.set(anchor=jnp.broadcast_to(anchor_init, (tree.topology.size, d)))
        tree = tree.at[tree.topology.is_leaf].set(anchor=leaf_obs)
    return tree


def init_continuous_tree(
    tree: Tree,
    leaf_obs: jax.Array,
    obs_var: float | jax.Array,
    *,
    d: int,
    n_steps: int,
    root_val: jax.Array | None = None,
    anchor_init: jax.Array | None = None,
) -> Tree:
    """Seed a continuous-edge BFFG tree with leaf observations and initial anchors.

    Writes leaves' **vertex** canonical message ``(prec_v, ptnl_v, log_norm)``
    from the isotropic-Gaussian likelihood :math:`y \\sim \\mathcal N(x,
    \\tau^2 I)` and seeds both ``anchor`` (leaves at observation, others at
    ``anchor_init``) and ``anchor_pa``. The per-edge ``precs`` / ``ptnls``
    trajectories are filled by :func:`continuous_bf_sweep`. Optionally pins
    the root trajectory.

    Args:
        tree: Empty tree with the schema returned by :func:`continuous_schema`.
            ``edge_len`` should already be set on each non-root node.
        leaf_obs: ``(n_leaves, d)`` leaf observations in BFS leaf order.
        obs_var: Scalar observation variance :math:`\\tau^2`.
        d: State dimension.
        n_steps: Number of substeps per edge (must match the schema).
        root_val: Optional ``(d,)`` value to pin at the root (broadcast to the
            root's full ``(n_steps + 1, d)`` ``vals`` trajectory).
        anchor_init: Optional ``(d,)`` initial anchor for non-leaf nodes (and
            for every node's ``anchor_pa`` before refinement). Defaults to
            ``root_val`` if given, else zeros.

    Returns:
        Tree with leaf vertex messages set, both anchors seeded, and
        (optionally) the root trajectory pinned.
    """
    msgs = _canonical_leaf_messages(leaf_obs, obs_var, d=d)
    # Leaf observation = vertex message (terminal/initial condition for the
    # edge backward filter). The per-edge trajectory is filled by the up-sweep.
    tree = tree.at[tree.topology.is_leaf].set(
        prec_v=msgs["prec"], ptnl_v=msgs["ptnl"], log_norm=msgs["log_norm"]
    )
    if root_val is not None:
        root_val = jnp.asarray(root_val)
        if "vals" in tree.schema:
            tree = tree.at[tree.topology.is_root].set(
                vals=jnp.broadcast_to(root_val, (n_steps + 1, d))
            )
        else:
            raise ValueError("Schema must declare 'vals' field to set root_val.")
    # Seed every node's two anchors (Algorithm 3 §7.1 linearisation points).
    # - anchor (child end of incoming edge):
    #     leaves -> leaf_obs (= posterior mean at leaf vertex)
    #     inner/root -> anchor_init (default = root_val)
    # - anchor_pa (parent end of incoming edge): anchor_init everywhere
    #   for the first BF; subsequent refine_anchor sweeps overwrite it with
    #   the parent's just-refined anchor.
    if "anchor" in tree.schema:
        if anchor_init is None:
            anchor_init = (
                jnp.zeros(d, dtype=leaf_obs.dtype) if root_val is None else jnp.asarray(root_val)
            )
        anchor_init = jnp.asarray(anchor_init)
        tree = tree.set(anchor=jnp.broadcast_to(anchor_init, (tree.topology.size, d)))
        tree = tree.at[tree.topology.is_leaf].set(anchor=leaf_obs)
        if "anchor_pa" in tree.schema:
            tree = tree.set(anchor_pa=jnp.broadcast_to(anchor_init, (tree.topology.size, d)))
    return tree


def _discrete_backward_filtering(
    prec,
    ptnl,
    log_norm,
    anchor,
    prxy_scale_fn,
    prxy_shift_fn,
    prxy_covar_fn,
    params,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One BF edge step. Returns ``(prec_msg, ptnl_msg, log_norm_msg)`` such
    that the parent's canonical message is the sum of these contributions
    over its children.

    The auxiliary transition ``x_v | x_pa ~ N(B x_pa + β, Q)`` is evaluated
    at the per-edge linearisation point ``anchor`` (Algorithm 3 §7.1).
    For linear-Gaussian models the user simply ignores ``anchor``.

    With child canonical message ``exp(c_v + F_v·x - ½x'H_v x)``, the
    integral ``∫ exp(...) N(x_v; Bx_pa+β, Q) dx_v`` gives a Gaussian
    canonical message in ``x_pa`` with:

    - ``prec_msg = Bᵀ (I + H_v Q)⁻¹ H_v B``
    - ``ptnl_msg = Bᵀ (I + H_v Q)⁻¹ (F_v - H_v β)``
    - ``log_norm_msg = c_v - ½ log det(I + H_v Q) + ½ F_v' Q (I+H_v Q)⁻¹ F_v
                       + F_v' Q (I+H_v Q)⁻¹ Q⁻¹ β - ½ β' (I+H_v Q)⁻¹ H_v β``
    """
    B, beta = prxy_scale_fn(anchor, params), prxy_shift_fn(anchor, params)
    Q = prxy_covar_fn(anchor, params)
    d = prec.shape[-1]
    phi_inv = jnp.eye(d, dtype=prec.dtype) + prec @ Q  # I + HQ
    Cinv = jnp.linalg.solve(phi_inv, prec)  # phi_inv⁻¹ H = C⁻¹
    prec_msg = B.T @ Cinv @ B
    ptnl_msg = B.T @ solve(phi_inv, ptnl - prec @ beta)

    # c-update:  Δc = -½ log det(phi_inv) + ½ F' Q phi_inv⁻¹ F
    #                 + F' Q phi_inv⁻¹ Q⁻¹ β - ½ β' phi_inv⁻¹ H β.
    sign, logdet_phi = jnp.linalg.slogdet(phi_inv)
    Q_phi_inv = jnp.linalg.solve(phi_inv.T, Q.T).T  # Q · phi_inv⁻¹
    term_F = 0.5 * ptnl @ Q_phi_inv @ ptnl
    term_F_beta = ptnl @ Q_phi_inv @ jnp.linalg.solve(Q, beta)
    term_beta = -0.5 * beta @ jnp.linalg.solve(phi_inv, prec @ beta)
    log_norm_msg = log_norm - 0.5 * logdet_phi + term_F + term_F_beta + term_beta
    return prec_msg, ptnl_msg, log_norm_msg


def _discrete_forward_guiding(
    x_pa,
    z,
    prec,
    ptnl,
    anchor,
    *,
    mean_fn,
    covar_fn,
    prxy_scale_fn,
    prxy_shift_fn,
    prxy_covar_fn,
    params,
) -> tuple[jax.Array, jax.Array]:
    d = prec.shape[-1]
    mu = mean_fn(x_pa, params)
    Q_true = covar_fn(x_pa, params)

    # Sample under P°: Nᶜᵃⁿ(F + Q⁻¹μ, H + Q⁻¹)  (Theorem 14).
    inv = jnp.linalg.solve(jnp.eye(d, dtype=prec.dtype) + Q_true @ prec, Q_true)  # (H + Q⁻¹)⁻¹
    m_ch = dot(inv, ptnl + solve(Q_true, mu))
    x_ch = m_ch + dot(cholesky(inv, lower=True, check_finite=False), z)

    # Weight w = (Pg)/(P̃g) = φ(H⁻¹F; μ, Q+H⁻¹) / φ(H⁻¹F; Φx+β, Q̃+H⁻¹).
    # Auxiliary is evaluated at the per-edge linearisation point `anchor`.
    H_inv = cho_solve(
        (cholesky(prec, lower=True, check_finite=False), True),
        jnp.eye(d, dtype=prec.dtype),
    )
    m_star = solve(prec, ptnl)  # H⁻¹F
    B, beta = prxy_scale_fn(anchor, params), prxy_shift_fn(anchor, params)
    mu_prxy = B @ x_pa + beta

    C_true = Q_true + H_inv
    C_prxy = prxy_covar_fn(anchor, params) + H_inv
    L_true = cholesky(C_true, lower=True, check_finite=False)
    L_prxy = cholesky(C_prxy, lower=True, check_finite=False)
    y_true = solve_triangular(L_true, m_star - mu, lower=True)
    y_prxy = solve_triangular(L_prxy, m_star - mu_prxy, lower=True)
    logdet_true = 2.0 * jnp.sum(jnp.log(jnp.diag(L_true)))
    logdet_prxy = 2.0 * jnp.sum(jnp.log(jnp.diag(L_prxy)))
    log_corr = -0.5 * (logdet_true - logdet_prxy) - 0.5 * (jnp.sum(y_true**2) - jnp.sum(y_prxy**2))
    return x_ch, log_corr


def discrete_bf_sweep(prxy_scale_fn, prxy_shift_fn, prxy_covar_fn) -> SweepFn:
    """Build the discrete-edge backward-filtering up-sweep (Theorem 14 §6.1).

    At each non-leaf parent, pulls each child's canonical message back through
    the linear-Gaussian auxiliary :math:`\\tilde P(x, dy) = \\mathcal N(y;
    \\Phi x + \\beta, Q)\\,dy` and fuses (sums) the contributions. The
    auxiliary is evaluated per-child at the child's ``anchor``.

    Args:
        prxy_scale_fn: Callable ``(anchor, params) -> (d, d) array`` returning
            the auxiliary scale matrix :math:`\\Phi`.
        prxy_shift_fn: Callable ``(anchor, params) -> (d,) array`` returning
            the auxiliary shift vector :math:`\\beta`.
        prxy_covar_fn: Callable ``(anchor, params) -> (d, d) SPD array``
            returning the auxiliary covariance :math:`Q`.

    Returns:
        A :class:`hyperiax.SweepFn` that, when applied to a tree, writes
        ``(prec, ptnl, log_norm)`` at every non-leaf node.

    Notes:
        Linear-Gaussian models (auxiliary equal to truth) ignore ``anchor``;
        for nonlinear models, iterate this sweep with
        :func:`discrete_refine_anchor` (Algorithm 3 §7.1).
    """

    @up(
        reads_children=("prec", "ptnl", "log_norm", "anchor"),
        writes=("prec", "ptnl", "log_norm"),
    )
    def _sweep(node, children, params):

        def per_child(child):
            prec_msg, ptnl_msg, log_norm_msg = _discrete_backward_filtering(
                child.prec,
                child.ptnl,
                child.log_norm,
                child.anchor,
                prxy_scale_fn,
                prxy_shift_fn,
                prxy_covar_fn,
                params,
            )
            return {"prec": prec_msg, "ptnl": ptnl_msg, "log_norm": log_norm_msg}

        msgs = children.map(per_child)
        # Canonical messages multiply at a parent: (prec, ptnl, log_norm) all sum.
        return {
            "prec": msgs.prec.sum(0),
            "ptnl": msgs.ptnl.sum(0),
            "log_norm": msgs.log_norm.sum(0),
        }

    return _sweep


def discrete_forward_sweep(mean_fn, covar_fn) -> SweepFn:
    """Build the unconditional forward-sampling down-sweep.

    Draws each non-root state from the true 1-step Gaussian kernel
    :math:`X_t \\mid X_{pa} = x \\sim \\mathcal N(\\mu(x), Q(x))` using the
    pre-stored noise :math:`z \\sim \\mathcal N(0, I)` in ``node.z``::

        val = mean_fn(parent.val, params)
              + chol(covar_fn(parent.val, params)) @ node.z

    Args:
        mean_fn: Callable ``(x_parent, params) -> (d,) array`` returning the
            true conditional mean :math:`\\mu(x)`.
        covar_fn: Callable ``(x_parent, params) -> (d, d) SPD array``
            returning the true conditional covariance :math:`Q(x)`.

    Returns:
        A :class:`hyperiax.SweepFn` that writes ``val`` at every non-root node.
        The root's ``val`` must be set by the caller (typically via
        :func:`init_discrete_tree`'s ``root_val``).
    """

    @down(
        reads_parent=("val",),
        reads=("z",),
        writes=("val",),
    )
    def _sweep(node, parent, params):
        covar = covar_fn(parent.val, params)
        return {
            "val": mean_fn(parent.val, params)
            + dot(cholesky(covar, lower=True, check_finite=False), node.z)
        }

    return _sweep


def discrete_fg_sweep(mean_fn, covar_fn, prxy_scale_fn, prxy_shift_fn, prxy_covar_fn) -> SweepFn:
    """Build the discrete-edge forward-guided down-sweep (Theorem 14 §6.1).

    For each non-root node, samples the guided proposal :math:`X_t^\\circ
    \\sim \\mathcal N^{\\mathrm{can}}(F + Q(x)^{-1}\\mu(x),\\ H + Q(x)^{-1})`
    using the parent state ``parent.val``, the node's canonical message
    ``(prec, ptnl)``, the true kernel ``(mean_fn, covar_fn)``, and the noise
    ``node.z``. Writes the importance log-weight

    .. math::

       \\log w = \\log\\varphi(H^{-1}F;\\ \\mu(x),\\ Q(x)+H^{-1})
                - \\log\\varphi(H^{-1}F;\\ \\Phi x+\\beta,\\ Q_{\\mathrm{aux}}+H^{-1})

    to ``log_corr`` (Theorem 14 step 3). Sum ``log_corr`` over non-root nodes
    to get the path's importance correction.

    Args:
        mean_fn / covar_fn: True kernel ``(x_parent, params) -> array``.
        prxy_scale_fn / prxy_shift_fn / prxy_covar_fn: Auxiliary kernel
            ``(anchor, params) -> array`` (same callables fed to
            :func:`discrete_bf_sweep`).

    Returns:
        A :class:`hyperiax.SweepFn` that writes ``(val, log_corr)`` at every
        non-root node. Run :func:`discrete_bf_sweep` first to populate the
        canonical messages.
    """

    @down(
        reads_parent=("val",),
        reads=("z", "prec", "ptnl", "anchor"),
        writes=("val", "log_corr"),
    )
    def _sweep(node, parent, params):
        x_ch, log_corr = _discrete_forward_guiding(
            parent.val,
            node.z,
            node.prec,
            node.ptnl,
            node.anchor,
            mean_fn=mean_fn,
            covar_fn=covar_fn,
            prxy_scale_fn=prxy_scale_fn,
            prxy_shift_fn=prxy_shift_fn,
            prxy_covar_fn=prxy_covar_fn,
            params=params,
        )
        return {"val": x_ch, "log_corr": log_corr}

    return _sweep


def discrete_refine_anchor() -> SweepFn:
    """Build the discrete anchor-refinement down-sweep (Algorithm 3 §7.1).

    Overwrites each non-root node's ``anchor`` with the BFFG-implied posterior
    mean :math:`H^{-1} F` derived from the canonical message
    ``(prec, ptnl)`` left by :func:`discrete_bf_sweep`. The root keeps its
    initial anchor (it is pinned at ``root_val``).

    Returns:
        A :class:`hyperiax.SweepFn` that writes ``anchor`` at every non-root
        node. Typical usage interleaves with the backward filter for several
        iterations until the anchor — and thus the auxiliary linearisation —
        converges::

            for _ in range(n_lin_iters):
                tree = bf(tree, params=theta)
                tree = refine(tree, params=theta)
    """

    @down(reads=("prec", "ptnl"), writes=("anchor",))
    def _sweep(node, parent, params):
        anchor = jnp.linalg.solve(node.prec, node.ptnl)
        return {"anchor": anchor}

    return _sweep


def _continuous_backward_filtering(
    ts,
    prec,
    ptnl,
    log_norm,
    anchor_pa,
    anchor_ch,
    *,
    prxy_diffusion_fn,
    prxy_scale_fn,
    prxy_shift_fn,
    params,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Backward-filter one edge. Returns ``(precs, ptnls, log_norm_at_pa)``
    where ``precs`` and ``ptnls`` are the per-step canonical trajectories
    and ``log_norm_at_pa`` is the canonical-message log-norm propagated to
    the parent end of the edge.

    Two anchor points define the auxiliary diffusion's linearisation along
    the edge (Algorithm 3 §7.1):

    - ``anchor_pa``: state at the parent end of the edge (t = ts[0]).
    - ``anchor_ch``: state at the child end of the edge (t = ts[-1]).

    The auxiliary diffusion ``ã(t)`` is linearly interpolated between
    ``σ̃(anchor_pa)σ̃ᵀ`` and ``σ̃(anchor_ch)σ̃ᵀ``.
    """
    if prxy_scale_fn is None and prxy_shift_fn is None:
        return _continuous_bf_anlt(
            ts,
            prec,
            ptnl,
            log_norm,
            anchor_pa,
            anchor_ch,
            prxy_scale_fn=None,
            prxy_shift_fn=None,
            prxy_diffusion_fn=prxy_diffusion_fn,
            params=params,
        )
    else:
        return _continuous_bf_ode(
            ts,
            prec,
            ptnl,
            log_norm,
            anchor_pa,
            anchor_ch,
            prxy_scale_fn=prxy_scale_fn,
            prxy_shift_fn=prxy_shift_fn,
            prxy_diffusion_fn=prxy_diffusion_fn,
            params=params,
        )


def _continuous_bf_anlt(
    ts,
    prec,
    ptnl,
    log_norm,
    anchor_pa,
    anchor_ch,
    *,
    prxy_scale_fn,
    prxy_shift_fn,
    prxy_diffusion_fn,
    params,
):
    # B = β = 0: driftless linear auxiliary with linearly-interpolated σ̃.
    # ã(t) = (t/T) ã_ch + (1 - t/T) ã_pa, where ã_x = σ̃(x) σ̃(x)ᵀ.
    # Then Φ̃⁻¹(t) = I + H_T · ∫_t^T ã(s) ds is computed analytically.
    sigma_pa = prxy_diffusion_fn(ts[0], anchor_pa, params)
    sigma_ch = prxy_diffusion_fn(ts[-1], anchor_ch, params)
    a_pa = sigma_pa @ sigma_pa.T
    a_ch = sigma_ch @ sigma_ch.T
    t1 = ts[-1]
    d = prec.shape[-1]

    # Hoist H_T · ã_pa and H_T · ã_ch once so per_t reduces to elementwise
    # weighting + one (d, d) solve per step.
    H_a_pa = prec @ a_pa
    H_a_ch = prec @ a_ch

    def per_t(t):
        # ∫_t^T ã(s) ds = c_ch · ã_ch + c_pa · ã_pa where
        #   c_ch = ∫_t^T (s/T) ds = (T² - t²) / (2T)
        #   c_pa = ∫_t^T (1 - s/T) ds = (T - t)² / (2T)
        c_ch = -(t**2 - t1**2) / (2.0 * t1)
        c_pa = ((t1 - t) ** 2) / (2.0 * t1)
        phi_inv = jnp.eye(d, dtype=prec.dtype) + c_ch * H_a_ch + c_pa * H_a_pa
        prec_t = jnp.linalg.solve(phi_inv, prec)
        ptnl_t = solve(phi_inv, ptnl)
        return prec_t, ptnl_t

    precs, ptnls = jax.vmap(per_t)(ts)

    # log-norm at the parent end (t = ts[0]).  Effective covariance over the
    # whole edge under linear ã interpolation is (T/2)(ã_pa + ã_ch).
    covar_eff = 0.5 * t1 * (a_pa + a_ch)
    phi_inv_full = jnp.eye(d, dtype=prec.dtype) + prec @ covar_eff  # I + HQ
    _sign, logdet_full = jnp.linalg.slogdet(phi_inv_full)
    # Quadratic term ½ Fᵀ M⁻¹ F with M = H + Q⁻¹, and M⁻¹ = Q (I + HQ)⁻¹.
    # The factor ordering matters: (I+HQ)⁻¹Q is *not* symmetric and only
    # equals Q(I+HQ)⁻¹ when H and Q commute (e.g. an isotropic leaf prec).
    # At internal edges the fused precision does not commute with ã, so the
    # message normaliser must use Q(I+HQ)⁻¹ — otherwise log h(x_root) grows
    # spuriously with the diffusion scale.
    quad = 0.5 * ptnl @ (covar_eff @ jnp.linalg.solve(phi_inv_full, ptnl))
    log_norm_at_pa = log_norm - 0.5 * logdet_full + quad
    return precs, ptnls, log_norm_at_pa


def _continuous_bf_ode(
    ts,
    prec,
    ptnl,
    log_norm,
    anchor_pa,
    anchor_ch,
    *,
    prxy_scale_fn,
    prxy_shift_fn,
    prxy_diffusion_fn,
    params,
):
    t1 = ts[-1]
    d = prec.shape[-1]

    def anchor_at(t):
        # Linear interpolation between parent end (t=0) and child end (t=T).
        return (t / t1) * anchor_ch + (1.0 - t / t1) * anchor_pa

    def vector_field(tau, y, args):
        # Recover (H, F, c) from the packed state.
        Ht = y[: d * d].reshape((d, d))
        Ft = y[d * d : d * d + d]
        # ct = y[d*d+d]  # not used in the RHS, just propagated
        t = t1 - tau  # integrate backward in time
        anchor_t = anchor_at(t)
        Bt = prxy_scale_fn(t, anchor_t, args)
        betat = prxy_shift_fn(t, anchor_t, args)
        sigmat = prxy_diffusion_fn(t, anchor_t, args)
        at = sigmat @ sigmat.T

        # Forward-in-t derivatives (Theorem 23, eq 29; ã = σ̃ σ̃ᵀ).
        dH = -Bt.T @ Ht - Ht @ Bt + Ht @ at @ Ht
        dF = -Bt.T @ Ft + Ht @ at @ Ft + Ht @ betat
        # Matching c-ODE (Kolmogorov backward, constant-term collection):
        #   dc/dt = -β'F + ½ tr(aH) - ½ F'aF.
        dc = -betat @ Ft + 0.5 * jnp.trace(at @ Ht) - 0.5 * Ft @ at @ Ft
        # Integrate in τ = T - t, so dy/dτ = -dy/dt.
        return -jnp.concatenate([dH.flatten(), dF.flatten(), jnp.asarray(dc).reshape(1)])

    y0 = jnp.concatenate([prec.flatten(), ptnl.flatten(), jnp.asarray(log_norm).reshape(1)])
    sol = solve_ode(vector_field, y0, ts, solver=RK4(), args=params)
    sol = sol[::-1]
    precs = sol[:, : d * d].reshape((-1, d, d))
    ptnls = sol[:, d * d : d * d + d].reshape((-1, d))
    log_norms = sol[:, d * d + d]
    return precs, ptnls, log_norms[0]  # parent end (after reversal)


def _continuous_forward_guiding(
    x_pa,
    ts,
    dws,
    precs,
    ptnls,
    anchor_pa,
    anchor_ch,
    drift_fn,
    diffusion_fn,
    prxy_scale_fn,
    prxy_shift_fn,
    prxy_diffusion_fn,
    params,
) -> tuple[jax.Array, jax.Array]:
    # Auxiliary's σ̃ varies linearly between anchor_pa (t=0) and anchor_ch (t=T)
    # to match the BF's interpolated ã. We pre-compute the endpoint ã's once;
    # each step reads ã(t_i) by linear lerp — same machinery the BF used.
    t1 = ts[-1]
    sigma_pa = prxy_diffusion_fn(ts[0], anchor_pa, params)
    sigma_ch = prxy_diffusion_fn(t1, anchor_ch, params)
    a_pa_endpoint = sigma_pa @ sigma_pa.T
    a_ch_endpoint = sigma_ch @ sigma_ch.T

    def anchor_at(t):
        return (t / t1) * anchor_ch + (1.0 - t / t1) * anchor_pa

    # Build the auxiliary drift closure. Both B(t) and β(t) see the
    # interpolated anchor — matches the BF's `anchor_at(t)` view.
    if prxy_scale_fn is None and prxy_shift_fn is None:
        prxy_drift_fn = lambda t, x, params: jnp.zeros_like(x)
    else:
        prxy_drift_fn = lambda t, x, params: (
            prxy_shift_fn(t, anchor_at(t), params) + prxy_scale_fn(t, anchor_at(t), params) @ x
        )

    def bridge_step_body(carry, val):
        i, x, log_corr = carry
        dt, dw = val
        t = ts[i]
        Ht, Ft = precs[i], ptnls[i]
        r = Ft - dot(Ht, x)
        f_true, g_true = drift_fn(t, x, params), diffusion_fn(t, x, params)
        f_prxy = prxy_drift_fn(t, x, params)
        a_true = g_true @ g_true.T
        # Auxiliary ã(t) — linearly interpolated between endpoint covariances.
        a_prxy_t = (t / t1) * a_ch_endpoint + (1.0 - t / t1) * a_pa_endpoint
        x_next = x + (f_true + dot(a_true, r)) * dt + dot(g_true, dw)

        # log w increment = (L - L̃)g / g   (Theorem 23 eq 32, Remark 24):
        #   (b - b̃)·r - ½ tr((a - ã) H) + ½ r'(a - ã) r.
        covar_diff = a_true - a_prxy_t  # a - ã (interpolated)
        drift_diff = f_true - f_prxy  # b - b̃
        log_corr_next = (
            log_corr
            + (jnp.dot(drift_diff, r) - 0.5 * jnp.sum(covar_diff * Ht) + 0.5 * (r @ covar_diff @ r))
            * dt
        )
        return (i + 1, x_next, log_corr_next), x

    dts = jnp.diff(ts)
    log_corr0 = jnp.zeros((), dtype=x_pa.dtype)
    (_, x_ch, log_corr), xs = jax.lax.scan(bridge_step_body, (0, x_pa, log_corr0), (dts, dws))
    return jnp.vstack((xs, x_ch)), log_corr


def continuous_bf_sweep(n_steps, prxy_scale_fn, prxy_shift_fn, prxy_diffusion_fn) -> SweepFn:
    """Build the continuous-edge backward-filtering up-sweep (Theorem 23 §7.1).

    For each parent, integrates the auxiliary-SDE backward equation (eq 29)
    over every child's edge — either analytically (closed form when
    :math:`B = \\beta = 0`) or via :class:`hyperiax.utils.ode.RK4` — to
    produce the per-step canonical ``(H, F)`` trajectory. Each edge is
    linearised between two endpoint anchors (``anchor_pa`` at :math:`t = 0`
    and ``anchor`` at :math:`t = \\tau_e`), with :math:`\\tilde a(t) =
    \\tilde\\sigma\\tilde\\sigma^\\top` linearly interpolated. The trajectory
    is scattered to the child (``writes_children``); the time-0 messages are
    fused into the parent's vertex message ``(prec_v, ptnl_v, log_norm)``.

    Args:
        n_steps: Number of integration substeps per edge.
        prxy_scale_fn: ``(t, anchor, params) -> (d, d)`` returning :math:`B(t)`,
            or ``None`` to take the driftless analytic path (requires
            ``prxy_shift_fn`` also ``None``).
        prxy_shift_fn: ``(t, anchor, params) -> (d,)`` returning
            :math:`\\beta(t)`, or ``None`` for the analytic path.
        prxy_diffusion_fn: ``(t, anchor, params) -> (d, d)`` returning
            :math:`\\tilde\\sigma(t)`; the auxiliary covariance is
            :math:`\\tilde a = \\tilde\\sigma \\tilde\\sigma^\\top`.

    Returns:
        A :class:`hyperiax.SweepFn` that writes ``(prec_v, ptnl_v, log_norm)``
        at every non-leaf node and the full ``(precs, ptnls)`` trajectories
        at every non-root node.
    """

    @up(
        reads_children=("edge_len", "prec_v", "ptnl_v", "log_norm", "anchor", "anchor_pa"),
        writes=("prec_v", "ptnl_v", "log_norm"),
        writes_children=("precs", "ptnls"),
    )
    def _sweep(node, children, params):

        def per_child(child):
            ts = jnp.linspace(0.0, child.edge_len, n_steps + 1, dtype=child.edge_len.dtype)
            # Two-anchor linearisation along each edge:
            #   anchor_pa = child.anchor_pa (parent end, t = 0)
            #   anchor_ch = child.anchor    (child end,  t = T)
            precs, ptnls, log_norm_at_pa = _continuous_backward_filtering(
                ts,
                child.prec_v,
                child.ptnl_v,
                child.log_norm,
                child.anchor_pa,
                child.anchor,
                prxy_diffusion_fn=prxy_diffusion_fn,
                prxy_scale_fn=prxy_scale_fn,
                prxy_shift_fn=prxy_shift_fn,
                params=params,
            )
            # Return the full trajectory (cached on the child edge for forward
            # guiding), its parent-end (time-0) message for the fusion, and the
            # canonical-message log-norm at the parent end. Explicit fields here
            # avoid indexing a segment-axis object after the map.
            return {
                "precs": precs,
                "ptnls": ptnls,
                "m_prec": precs[0],
                "m_ptnl": ptnls[0],
                "m_log_norm": log_norm_at_pa,
            }

        # Per child: backward-filter its edge from the child's vertex message
        # (terminal) into a full (n_steps+1) (H, F) trajectory.
        msgs = children.map(per_child)
        # Fusion: canonical messages multiply at the parent, so (prec_v, ptnl_v,
        # log_norm) all sum over children. The result is the terminal/initial
        # condition for this node's own edge, one level up.
        return {
            "precs": msgs.precs,  # (n_steps+1, d, d) per child -> children
            "ptnls": msgs.ptnls,  # (n_steps+1, d)    per child -> children
            "prec_v": msgs.m_prec.sum(0),  # (d, d)                      -> this node
            "ptnl_v": msgs.m_ptnl.sum(0),  # (d,)
            "log_norm": msgs.m_log_norm.sum(0),  # ()
        }

    return _sweep


def continuous_forward_sweep(n_steps, drift_fn, diffusion_fn) -> SweepFn:
    """Build the unconditional SDE forward-sampling down-sweep.

    For each non-root node, integrates the true SDE
    :math:`dX_u = b(u, X_u)\\,du + \\sigma(u, X_u)\\,dW_u` from the parent's
    terminal value ``parent.vals[-1]`` over the edge using Euler-Maruyama
    and the pre-stored increments :math:`dW = \\sqrt{\\Delta t}\\, z` derived
    from ``node.zs``.

    Args:
        n_steps: Number of Euler-Maruyama substeps per edge.
        drift_fn: ``(t, x, params) -> (d,)`` returning the true drift.
        diffusion_fn: ``(t, x, params) -> (d, noise)`` returning the true
            diffusion matrix :math:`\\sigma(t, x)`.

    Returns:
        A :class:`hyperiax.SweepFn` that writes the full per-edge trajectory
        to ``vals`` at every non-root node. The root's ``vals`` must be set by
        the caller (typically via :func:`init_continuous_tree`'s ``root_val``).
    """

    @down(
        reads_parent=("vals",),
        reads=("zs", "edge_len"),
        writes=("vals",),
    )
    def _sweep(node, parent, params):
        ts = jnp.linspace(0, node.edge_len, n_steps + 1, dtype=node.edge_len.dtype)
        dws = jnp.sqrt(jnp.diff(ts))[:, None] * node.zs
        x0 = parent.vals[-1]
        ys = solve_sde(drift_fn, diffusion_fn, x0, ts, dws, solver=EulerMaruyama(), args=params)
        return {"vals": ys}

    return _sweep


def continuous_fg_sweep(
    n_steps, drift_fn, diffusion_fn, prxy_scale_fn, prxy_shift_fn, prxy_diffusion_fn
) -> SweepFn:
    """Build the continuous-edge forward-guided down-sweep (Theorem 23 §7.1).

    For each non-root node, integrates the guided SDE (eq 31)

    .. math::

       dX_u^\\circ = \\big(b(u, X_u^\\circ) + a(u, X_u^\\circ)(F(u) - H(u)X_u^\\circ)\\big)du
                     + \\sigma(u, X_u^\\circ)\\,dW_u

    from ``parent.vals[-1]`` over the edge, using the BF-cached
    ``(precs, ptnls)`` trajectory as the guiding term and ``node.zs`` for the
    noise. Writes the per-step Theorem-23 importance log-weight increment

    .. math::

       \\frac{(\\mathcal L - \\tilde{\\mathcal L})g}{g}
         = (b - \\tilde b)\\cdot r - \\tfrac12 \\mathrm{tr}((a - \\tilde a) H)
           + \\tfrac12 r^\\top(a - \\tilde a) r,
         \\qquad r = F - H X^\\circ,

    integrated over the edge into ``log_corr`` (Remark 24). Auxiliary
    :math:`\\tilde a(t)` is linearly interpolated between the two anchors —
    same convention as :func:`continuous_bf_sweep`.

    Args:
        n_steps: Number of substeps per edge.
        drift_fn / diffusion_fn: True SDE ``(t, x, params) -> array``.
        prxy_scale_fn / prxy_shift_fn / prxy_diffusion_fn: Auxiliary linear-SDE
            ``(t, anchor, params) -> array`` (same callables fed to
            :func:`continuous_bf_sweep`; pass ``None`` to ``prxy_scale_fn``
            and ``prxy_shift_fn`` for the driftless analytic case).

    Returns:
        A :class:`hyperiax.SweepFn` that writes ``(vals, log_corr)`` at every
        non-root node. Run :func:`continuous_bf_sweep` first.
    """

    @down(
        reads_parent=("vals",),
        reads=("zs", "edge_len", "precs", "ptnls", "anchor", "anchor_pa"),
        writes=(
            "vals",
            "log_corr",
        ),
    )
    def _sweep(node, parent, params):
        ts = jnp.linspace(0, node.edge_len, n_steps + 1, dtype=node.edge_len.dtype)
        dws = jnp.sqrt(jnp.diff(ts))[:, None] * node.zs
        # Two-anchor linearisation: node.anchor_pa at edge start (t=0) and
        # node.anchor at edge end (t=T) — matches what the BF sweep used.
        xs, log_corr = _continuous_forward_guiding(
            parent.vals[-1],
            ts,
            dws,
            node.precs,
            node.ptnls,
            node.anchor_pa,
            node.anchor,
            drift_fn,
            diffusion_fn,
            prxy_scale_fn,
            prxy_shift_fn,
            prxy_diffusion_fn,
            params,
        )
        return {"vals": xs, "log_corr": log_corr}

    return _sweep


def continuous_refine_anchor() -> SweepFn:
    """Build the continuous anchor-refinement down-sweep (Algorithm 3 §7.1).

    Walks root → leaves, writing two fields per non-root node:

    - ``anchor = prec_v⁻¹ ptnl_v`` — the BFFG-implied posterior mean at THIS
      vertex (child end of the incoming edge).
    - ``anchor_pa = parent.anchor`` — the parent's just-refined anchor,
      cached on the child so the next up-sweep can read both endpoints inside
      ``children.map(...)`` without a segment-aware gather.

    Because ``@down`` proceeds top-down, ``parent.anchor`` reads the parent's
    UPDATED value (refined at the previous level of this sweep). The root
    keeps its initial anchor (it is pinned at ``root_val``).

    Returns:
        A :class:`hyperiax.SweepFn` that writes ``(anchor, anchor_pa)`` at
        every non-root node. Typical usage iterates with the BF up-sweep::

            for _ in range(n_lin_iters):
                tree = bf(tree, params=theta)
                tree = refine(tree, params=theta)
    """

    @down(
        reads=("prec_v", "ptnl_v"),
        reads_parent=("anchor",),
        writes=("anchor", "anchor_pa"),
    )
    def _sweep(node, parent, params):
        anchor = jnp.linalg.solve(node.prec_v, node.ptnl_v)
        return {"anchor": anchor, "anchor_pa": parent.anchor}

    return _sweep


def propagate_linearization(tree: Tree, params=None) -> Tree:
    """Propagate continuous BFFG linearisation anchors top-down.

    This is the public one-shot form of :func:`continuous_refine_anchor`.
    It overwrites each non-root node's ``anchor`` with ``prec_v^-1 ptnl_v``
    and caches the parent's just-refined anchor in ``anchor_pa``.

    Args:
        tree: Continuous BFFG tree after :func:`continuous_bf_sweep`.
        params: Optional params object passed through to the underlying sweep.

    Returns:
        Tree with updated ``anchor`` and ``anchor_pa`` fields.
    """
    return continuous_refine_anchor()(tree, params=params)
