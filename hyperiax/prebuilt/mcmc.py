"""MCMC building blocks: Metropolis-Hastings on arbitrary JAX pytrees.

State-agnostic by design: an MCMC state is just a pytree (a :class:`Tree`,
a dict, a tuple, nested combinations — anything :mod:`jax.tree_util`
recognizes). The engine does not distinguish "latent state" from
"hyperparameter"; both flow through the same machinery, and joint
inference is "make state a dict of both."

API
---
- :class:`MHState`: ``(position, log_target)`` carried through the chain so
  each step only needs one fresh log-density evaluation.
- :func:`init_state`: evaluate the log-target once at the starting position.
- :func:`metropolis_step`: one symmetric Metropolis-Hastings update.
- :func:`run_chain`: drive a chain of N kernel steps via :func:`jax.lax.scan`.
- :func:`random_walk_proposal`: Gaussian random walk applied uniformly to
  every pytree leaf.
- :func:`crank_nicolson_proposal`: preconditioned Crank-Nicolson proposer
  for Gaussian-prior states (preserves the prior, so the acceptance ratio
  collapses to the likelihood ratio).

Constraints (e.g. ``σ² > 0``) are deliberately not built in. Propose in
the unconstrained space (e.g. on ``log_sigma``), recover the constrained
value inside ``log_target_fn`` (``jnp.exp(log_sigma)``), and add the
change-of-variables Jacobian term yourself. See notebook 05 for a worked
example.
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

    def __repr__(self) -> str:
        return (
            f"MHState(log_target={float(self.log_target):.4f}, "
            f"position={type(self.position).__name__})"
        )


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
    """Drive a chain of ``n_steps`` kernel updates via :func:`jax.lax.scan`.

    The whole loop compiles to a single XLA program — calling
    ``run_chain`` repeatedly with the same kernel hits the JIT cache.

    Args:
        key: PRNG key.
        init: initial :class:`MHState`. Build with :func:`init_state`.
        kernel_fn: ``(key, state) -> (new_state, info)``. Common usage::

                from functools import partial
                kernel = partial(metropolis_step,
                                 propose_fn=random_walk_proposal(0.5),
                                 log_target_fn=log_target_fn)

            For Gibbs-like multi-block schemes, write your own::

                def kernel(key, state):
                    k1, k2 = jax.random.split(key)
                    state, i1 = metropolis_step(k1, state, prop_block_1, log_target_fn)
                    state, i2 = metropolis_step(k2, state, prop_block_2, log_target_fn)
                    return state, {"acc_1": i1["accepted"], "acc_2": i2["accepted"]}
        n_steps: number of kernel calls.
        savef: optional projection ``MHState -> small_pytree`` applied at
            every step before stacking into the trace. ``None`` (default)
            stacks the entire :class:`MHState`, which is fine for cheap
            positions but memory-heavy when the position contains big
            arrays (e.g. SDE noise fields). Pass e.g.
            ``savef=lambda s: s.position['log_params']`` to keep only the
            small subset you actually want to plot.

    Returns:
        ``(trace, info)``. With ``savef=None`` the trace is an
        :class:`MHState` whose leaves carry leading axis ``n_steps``. With
        a ``savef`` the trace is whatever ``savef`` returned, stacked
        along a leading axis of length ``n_steps``. ``info`` is the
        stacked dict of per-step diagnostics from the kernel; acceptance
        rate is ``float(info["accepted"].mean())``.

    Multi-chain via ``jax.vmap``:
        When ``jax_tqdm`` is installed, ``run_chain`` works under
        ``jax.vmap`` with one progress bar per chain. Wrap each chain's
        ``MHState`` with ``jax_tqdm.PBar(id=chain_id, carry=mh_state)``;
        ``run_chain`` itself needs no changes. Typical pattern::

            from jax_tqdm import PBar

            inits = jax.vmap(lambda x: init_state(x, log_target))(init_positions)
            init_pbars = jax.vmap(lambda i, c: PBar(id=i, carry=c))(
                jnp.arange(n_chains), inits,
            )
            keys = jax.random.split(rng, n_chains)
            run_multi = jax.jit(jax.vmap(
                lambda k, init: run_chain(k, init, kernel_fn, n_steps=N, savef=saver),
            ))
            traces, infos = run_multi(keys, init_pbars)

        Returned ``traces`` / ``infos`` carry an extra leading
        ``(n_chains,)`` axis. Each chain's progress bar is identified by
        the contiguous integer index in ``PBar.id``.
    """
    keys = jax.random.split(key, n_steps)
    save = savef if savef is not None else (lambda s: s)

    # jax_tqdm requires the scan xs to expose the iteration index as either
    # the whole x (scalar int) or the first element of a tuple. We always
    # pair ``jnp.arange(n_steps)`` with the PRNG keys and unpack inside so
    # the body shape is the same whether or not the tqdm decoration applies.
    def body(state, x):
        _iter, key = x
        new_state, info = kernel_fn(key, state)
        return new_state, (save(new_state), info)

    if _HAS_JAX_TQDM:
        body = _scan_tqdm(n=n_steps, desc="Running MCMC chain")(body)

    xs = (jnp.arange(n_steps), keys)
    _, (trace, info) = jax.lax.scan(body, init, xs)
    return trace, info


# ── pre-built proposers ────────────────────────────────────────────
def random_walk_proposal(scale: float) -> Callable[[jax.Array, Any], Any]:
    """Gaussian random-walk proposer with uniform scale per pytree leaf.

    Each leaf ``x`` is perturbed independently: ``x' = x + scale · ε``,
    ``ε ~ N(0, I)``. For per-leaf scales, write your own ``propose_fn``.

    Args:
        scale: standard deviation of the additive Gaussian step.

    Returns:
        ``(key, position) -> new_position``.
    """
    def propose(key: jax.Array, position: Any) -> Any:
        leaves, treedef = jax.tree.flatten(position)
        keys = jax.random.split(key, max(1, len(leaves)))
        new_leaves = [
            x + scale * jax.random.normal(
                k, jnp.shape(x), dtype=jnp.result_type(x, jnp.float32)
            )
            for k, x in zip(keys, leaves)
        ]
        return jax.tree.unflatten(treedef, new_leaves)
    return propose


def crank_nicolson_proposal(beta: float) -> Callable[[jax.Array, Any], Any]:
    """Preconditioned Crank-Nicolson proposer for Gaussian-prior states.

    Updates each pytree leaf as

        x' = √(1 - β²) · x + β · ε,   ε ~ N(0, I),

    which preserves the standard-normal prior on ``x`` exactly. The
    Metropolis acceptance therefore reduces to the *likelihood* ratio —
    you do not need a separate prior or Jacobian term in your
    ``log_target_fn``. This is the standard mover for the latent noise
    field of an SDE bridge under BFFG.

    Args:
        beta: step size in ``[0, 1]``. ``β = 0`` is no move; ``β = 1`` is
            iid resampling. Typical good values are 0.01–0.1; tune for
            ~25-40% acceptance rate.

    Returns:
        ``(key, position) -> new_position``.

    Note:
        Assumes each leaf is a-priori standard normal with as many iid
        entries as its shape. For non-Gaussian priors, use
        :func:`random_walk_proposal` and include the prior in ``log_target_fn``.
    """
    scale_old = jnp.sqrt(1.0 - beta ** 2)

    def propose(key: jax.Array, position: Any) -> Any:
        leaves, treedef = jax.tree.flatten(position)
        keys = jax.random.split(key, max(1, len(leaves)))
        new_leaves = [
            scale_old * x + beta * jax.random.normal(
                k, jnp.shape(x), dtype=jnp.result_type(x, jnp.float32)
            )
            for k, x in zip(keys, leaves)
        ]
        return jax.tree.unflatten(treedef, new_leaves)
    return propose
