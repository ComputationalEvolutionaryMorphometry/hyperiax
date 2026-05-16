"""MCMC building blocks: Metropolis-Hastings on arbitrary JAX pytrees.

The state is any pytree the kernel can transport; the engine does not
distinguish latent state from hyperparameters. Constraints (e.g.
``σ² > 0``) are not built in — propose in the unconstrained space and
fold any change-of-variables Jacobian into ``log_target_fn``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

# Optional progress-bar decoration for :func:`run_chain`. ``jax_tqdm`` lives in
# the ``[dependency-groups] notebooks`` group; when it isn't installed (core /
# dev install only) the import gracefully degrades to a no-op so
# ``hyperiax.prebuilt.mcmc`` stays usable with just JAX.
try:
    from jax_tqdm import scan_tqdm as _scan_tqdm
    _HAS_JAX_TQDM = True
except ImportError:
    _HAS_JAX_TQDM = False
    _scan_tqdm = None  # type: ignore[assignment]


# ── chain state ────────────────────────────────────────────────────
@dataclass(frozen=True)
class MHState:
    """Cached Metropolis-Hastings state.

    Carries the current position alongside the value of the log target
    density there, so each :func:`metropolis_step` only triggers one
    fresh log-target evaluation.
    """

    position: Any
    log_target: jax.Array


# Register as a JAX pytree so it can ride through ``lax.scan`` as carry.
jax.tree_util.register_pytree_node(
    MHState,
    lambda s: ((s.position, s.log_target), None),
    lambda _aux, children: MHState(position=children[0], log_target=children[1]),
)


def init_state(
    position: Any,
    log_target_fn: Callable[[Any], jax.Array],
) -> MHState:
    """Build an :class:`MHState` by evaluating ``log_target_fn`` at ``position`` once."""
    return MHState(position=position, log_target=log_target_fn(position))


# ── core kernel ────────────────────────────────────────────────────
def metropolis_step(
    key: jax.Array,
    state: MHState,
    propose_fn: Callable[[jax.Array, Any], Any],
    log_target_fn: Callable[[Any], jax.Array],
) -> tuple[MHState, dict]:
    """One symmetric Metropolis-Hastings update.

    Symmetric proposals only — for asymmetric ``q(·|·)``, either build the
    log-q ratio into ``log_target_fn`` or write a custom kernel.

    Args:
        key: PRNG key for this step.
        state: current :class:`MHState`.
        propose_fn: ``(key, position) -> new_position`` — symmetric proposal
            on the position pytree. Standard pre-built choices:
            :func:`random_walk_proposal`, :func:`crank_nicolson_proposal`.
            For per-block proposals (Gibbs-like schemes), compose multiple
            calls to ``metropolis_step`` inside your own kernel.
        log_target_fn: ``position -> scalar log-density of the target``.
            The chain converges to this density times any change-of-variables
            Jacobian you fold in.

    Returns:
        ``(new_state, info)`` where ``info`` contains:

        - ``"accepted"``: bool — was the proposal accepted?
        - ``"log_alpha"``: scalar — the log acceptance ratio
          ``log π(proposed) − log π(current)``.
    """
    k_propose, k_accept = jax.random.split(key)
    proposed_position = propose_fn(k_propose, state.position)
    proposed_log_target = log_target_fn(proposed_position)

    log_alpha = proposed_log_target - state.log_target
    u = jax.random.uniform(k_accept)
    accepted = jnp.log(u) < log_alpha

    new_position = jax.tree.map(
        lambda p, c: jnp.where(accepted, p, c),
        proposed_position,
        state.position,
    )
    new_log_target = jnp.where(accepted, proposed_log_target, state.log_target)
    return (
        MHState(position=new_position, log_target=new_log_target),
        {"accepted": accepted, "log_alpha": log_alpha},
    )


# ── chain driver ───────────────────────────────────────────────────
def run_chain(
    key: jax.Array,
    init: MHState,
    kernel_fn: Callable[[jax.Array, MHState], tuple[MHState, dict]],
    n_steps: int,
    *,
    savef: Callable[[MHState], Any] | None = None,
) -> tuple[Any, dict]:
    """Drive ``n_steps`` kernel updates via :func:`jax.lax.scan`.

    Args:
        key: PRNG key.
        init: initial :class:`MHState`. Build with :func:`init_state`.
        kernel_fn: ``(key, state) -> (new_state, info)``. Typically
            ``partial(metropolis_step, propose_fn=..., log_target_fn=...)``;
            for Gibbs-like schemes, write a kernel that composes multiple
            ``metropolis_step`` calls.
        n_steps: number of kernel calls.
        savef: optional projection ``MHState -> small_pytree`` applied per
            step before stacking — keeps the trace small when the position
            contains big arrays. Default stacks the full :class:`MHState`.

    Returns:
        ``(trace, info)`` with leading axis ``n_steps``. Composes under
        ``jax.vmap`` for multi-chain runs; wrap each chain's state in
        ``jax_tqdm.PBar(id=i, carry=...)`` for per-chain progress bars.
    """
    keys = jax.random.split(key, n_steps)
    save = savef if savef is not None else (lambda s: s)

    # jax_tqdm reads the iteration index off the first element of xs;
    # we pair (arange, keys) so the body shape is unconditional.
    def body(state, x):
        _iter, key = x
        new_state, info = kernel_fn(key, state)
        return new_state, (save(new_state), info)

    if _HAS_JAX_TQDM:
        body = _scan_tqdm(n=n_steps, desc="Running MCMC chain", tqdm_type='std')(body)

    xs = (jnp.arange(n_steps), keys)
    _, (trace, info) = jax.lax.scan(body, init, xs)
    return trace, info


# ── pre-built proposers ────────────────────────────────────────────
def _per_leaf_proposal(
    leaf_step: Callable[[jax.Array, jax.Array], jax.Array],
) -> Callable[[jax.Array, Any], Any]:
    """Wrap a per-leaf ``(noise, x) -> x'`` rule into a pytree proposer."""
    def propose(key: jax.Array, position: Any) -> Any:
        leaves, treedef = jax.tree.flatten(position)
        keys = jax.random.split(key, max(1, len(leaves)))
        new_leaves = [
            leaf_step(
                jax.random.normal(
                    k, jnp.shape(x), dtype=jnp.result_type(x, jnp.float32)
                ),
                x,
            )
            for k, x in zip(keys, leaves)
        ]
        return jax.tree.unflatten(treedef, new_leaves)
    return propose


def random_walk_proposal(scale: float) -> Callable[[jax.Array, Any], Any]:
    """Gaussian random-walk proposer: per leaf ``x' = x + scale · ε``."""
    return _per_leaf_proposal(lambda eps, x: x + scale * eps)


def crank_nicolson_proposal(beta: float) -> Callable[[jax.Array, Any], Any]:
    """Preconditioned Crank-Nicolson proposer for standard-Gaussian-prior leaves.

    Per leaf ``x' = √(1 - β²) · x + β · ε``, which preserves the
    ``N(0, I)`` prior exactly — Metropolis acceptance then reduces to the
    pure likelihood ratio (no prior / Jacobian terms needed). Good ``β`` is
    0.01–0.1; tune for ~25–40% acceptance.
    """
    scale_old = jnp.sqrt(1.0 - beta ** 2)
    return _per_leaf_proposal(lambda eps, x: scale_old * x + beta * eps)
