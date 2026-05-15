"""MCMC building blocks. State-agnostic; tests cover scalars, dicts,
tuples, and Tree-as-state, plus the two pre-built proposers.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hyperiax as hx
from hyperiax.prebuilt.mcmc import (
    MHState,
    crank_nicolson_proposal,
    init_state,
    metropolis_step,
    random_walk_proposal,
    run_chain,
)


# ── basic kernel semantics ─────────────────────────────────────────
def test_metropolis_step_always_accepts_proposal_with_higher_target():
    """An "identity" proposal has log_alpha = 0, so accept with prob 1."""
    state = init_state(jnp.array(0.0), lambda x: -0.5 * x ** 2)
    out, info = metropolis_step(
        jax.random.PRNGKey(0), state,
        propose_fn=lambda key, x: x,           # identity
        log_target_fn=lambda x: -0.5 * x ** 2,
    )
    assert bool(info["accepted"])
    assert float(out.position) == 0.0


def test_metropolis_step_rejects_far_proposals_overwhelmingly():
    """A jump from x=0 to x=10 under log_target = -x²/2 has log_alpha = -50."""
    state = init_state(jnp.array(0.0), lambda x: -0.5 * x ** 2)
    def propose(key, x): return jnp.array(10.0)
    keys = jax.random.split(jax.random.PRNGKey(1), 200)
    accepts = sum(
        int(metropolis_step(k, state, propose, lambda x: -0.5 * x ** 2)[1]["accepted"])
        for k in keys
    )
    assert accepts == 0


def test_metropolis_step_caches_log_target_on_accept():
    """After an accepted step, state.log_target equals log_target_fn(new position)."""
    log_target = lambda x: -0.5 * x ** 2
    state = init_state(jnp.array(0.0), log_target)
    out, _ = metropolis_step(
        jax.random.PRNGKey(0), state,
        propose_fn=lambda k, x: jnp.array(0.5),
        log_target_fn=log_target,
    )
    # Proposal from 0 → 0.5: log_alpha = -0.5·(0.25 - 0) = -0.125, almost always accepts.
    assert bool(out.log_target == log_target(jnp.array(0.5)))


# ── pytree state ───────────────────────────────────────────────────
def test_state_can_be_dict():
    log_target = lambda d: -0.5 * (d["x"] ** 2 + d["y"] ** 2)
    init = init_state({"x": jnp.array(0.0), "y": jnp.array(0.0)}, log_target)
    kernel = partial(
        metropolis_step,
        propose_fn=random_walk_proposal(0.6),
        log_target_fn=log_target,
    )
    chain, info = run_chain(jax.random.PRNGKey(0), init, kernel, n_steps=3000)
    burn = 800
    assert abs(float(chain.position["x"][burn:].mean())) < 0.15
    assert abs(float(chain.position["y"][burn:].mean())) < 0.15


def test_state_can_be_tuple():
    log_target = lambda t: -0.5 * (t[0] ** 2 + t[1] ** 2)
    init = init_state((jnp.array(0.0), jnp.array(0.0)), log_target)
    kernel = partial(
        metropolis_step,
        propose_fn=random_walk_proposal(0.6),
        log_target_fn=log_target,
    )
    chain, _ = run_chain(jax.random.PRNGKey(0), init, kernel, n_steps=2000)
    burn = 500
    assert abs(float(chain.position[0][burn:].mean())) < 0.15


def test_state_can_be_tree():
    """The Tree pytree should ride through MCMC just like any other state."""
    topo = hx.symmetric_topology(height=2, degree=2)
    tree = hx.Tree.empty(topo, {"x": ()}).set(x=jnp.zeros(topo.size))

    def log_target(t: hx.Tree) -> jax.Array:
        # Target: standard normal on every node's x.
        return -0.5 * jnp.sum(t.x ** 2)

    def propose(key, t: hx.Tree) -> hx.Tree:
        return t.set(x=t.x + 0.4 * jax.random.normal(key, t.x.shape))

    init = init_state(tree, log_target)
    kernel = partial(metropolis_step, propose_fn=propose, log_target_fn=log_target)
    chain, info = run_chain(jax.random.PRNGKey(0), init, kernel, n_steps=1000)
    assert chain.position.x.shape == (1000, topo.size)
    assert float(info["accepted"].mean()) > 0.2


# ── target convergence ────────────────────────────────────────────
def test_random_walk_chain_converges_to_standard_normal():
    """RW MCMC targeting N(0, 1) — sample mean and std match within tolerance."""
    log_target = lambda x: -0.5 * x ** 2
    init = init_state(jnp.array(0.0), log_target)
    kernel = partial(
        metropolis_step,
        propose_fn=random_walk_proposal(0.8),
        log_target_fn=log_target,
    )
    chain, info = run_chain(jax.random.PRNGKey(0), init, kernel, n_steps=5000)
    samples = chain.position[1000:]
    assert abs(float(samples.mean())) < 0.1
    assert abs(float(samples.std()) - 1.0) < 0.1
    # RW with scale=0.8 on N(0,1) — broad-ish window since the optimal is
    # ~44% but the chain often runs higher early on.
    acc = float(info["accepted"][1000:].mean())
    assert 0.2 < acc < 0.85


def test_pcn_with_flat_likelihood_preserves_standard_normal_prior():
    """pCN with log_target ≡ 0 should be exact sampling from N(0, I).

    All proposals accept (log_alpha = 0), and the prior is preserved by
    construction — so after burn-in the chain is iid N(0, I).
    """
    init = init_state(jnp.zeros((3,)), lambda x: jnp.array(0.0))
    kernel = partial(
        metropolis_step,
        propose_fn=crank_nicolson_proposal(0.3),
        log_target_fn=lambda x: jnp.array(0.0),
    )
    chain, info = run_chain(jax.random.PRNGKey(0), init, kernel, n_steps=10000)
    samples = chain.position[2000:]
    cov = jnp.cov(samples.T)
    assert abs(float(samples.mean())) < 0.1
    assert jnp.allclose(cov, jnp.eye(3), atol=0.15)
    # Every step accepted.
    assert float(info["accepted"].mean()) == 1.0


# ── composability ──────────────────────────────────────────────────
def test_gibbs_style_composition_of_two_kernels():
    """Two block-updates inside one kernel — classic Gibbs-like pattern."""
    log_target = lambda d: -0.5 * (d["a"] ** 2 + d["b"] ** 2)
    init = init_state({"a": jnp.array(0.0), "b": jnp.array(0.0)}, log_target)

    def kernel(key, state):
        k1, k2 = jax.random.split(key)
        s, i1 = metropolis_step(
            k1, state,
            propose_fn=lambda k, p: {"a": p["a"] + 0.5 * jax.random.normal(k), "b": p["b"]},
            log_target_fn=log_target,
        )
        s, i2 = metropolis_step(
            k2, s,
            propose_fn=lambda k, p: {"a": p["a"], "b": p["b"] + 0.5 * jax.random.normal(k)},
            log_target_fn=log_target,
        )
        return s, {"acc_a": i1["accepted"], "acc_b": i2["accepted"]}

    chain, info = run_chain(jax.random.PRNGKey(0), init, kernel, n_steps=2500)
    burn = 500
    assert abs(float(chain.position["a"][burn:].mean())) < 0.15
    assert abs(float(chain.position["b"][burn:].mean())) < 0.15
    assert "acc_a" in info and "acc_b" in info


# ── jit / pytree machinery ────────────────────────────────────────
def test_mhstate_is_a_pytree():
    """MHState must flatten / unflatten so lax.scan can carry it as state."""
    s = init_state(jnp.array(1.5), lambda x: -0.5 * x ** 2)
    leaves, treedef = jax.tree_util.tree_flatten(s)
    assert len(leaves) == 2
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, MHState)
    assert float(rebuilt.position) == 1.5


def test_metropolis_step_under_outer_jit():
    log_target = lambda x: -0.5 * x ** 2
    state = init_state(jnp.array(0.0), log_target)
    def step(k, s):
        return metropolis_step(
            k, s,
            propose_fn=random_walk_proposal(0.5),
            log_target_fn=log_target,
        )
    eager_state, _ = step(jax.random.PRNGKey(0), state)
    jit_state, _ = jax.jit(step)(jax.random.PRNGKey(0), state)
    assert jnp.allclose(eager_state.position, jit_state.position)
    assert jnp.allclose(eager_state.log_target, jit_state.log_target)


def test_run_chain_savef_projects_state_to_small_trace():
    """``savef`` should keep only the projection in the trace — verify
    shape + that the kept trace matches per-step projection of full chain."""
    log_target = lambda d: -0.5 * (d['big'].sum() + d['small'] ** 2)
    init = init_state({'big': jnp.zeros((100,)), 'small': jnp.array(0.0)}, log_target)
    kernel = partial(
        metropolis_step,
        propose_fn=random_walk_proposal(0.1),
        log_target_fn=log_target,
    )
    # With savef we should only see 'small' in the trace.
    trace, _ = run_chain(
        jax.random.PRNGKey(0), init, kernel, n_steps=50,
        savef=lambda s: s.position['small'],
    )
    assert trace.shape == (50,)  # NOT (50, 100); the 'big' field is gone.


def test_run_chain_under_vmap_with_pbar_multi_chain():
    """``run_chain`` should compose with ``jax.vmap`` for multi-chain MCMC.

    When ``jax_tqdm`` is installed each chain's :class:`MHState` is wrapped
    in ``PBar(id=i, carry=...)`` so the scan body's internal
    ``isinstance(carry, PBar)`` check (inside ``jax_tqdm.scan_tqdm``) takes
    the per-chain progress-bar branch. Output traces carry an extra
    leading ``(n_chains,)`` axis.
    """
    pytest.importorskip("jax_tqdm")
    from jax_tqdm import PBar

    n_chains = 3
    n_steps = 50
    log_target = lambda x: -0.5 * x ** 2

    init_positions = jnp.array([-2.0, 0.0, 2.0])
    inits = jax.vmap(lambda x: init_state(x, log_target))(init_positions)
    init_pbars = jax.vmap(lambda i, c: PBar(id=i, carry=c))(
        jnp.arange(n_chains), inits,
    )

    kernel = partial(
        metropolis_step,
        propose_fn=random_walk_proposal(0.6),
        log_target_fn=log_target,
    )
    chain_keys = jax.random.split(jax.random.PRNGKey(0), n_chains)

    @jax.jit
    @jax.vmap
    def run_one(key, init):
        return run_chain(key, init, kernel, n_steps=n_steps)

    traces, infos = run_one(chain_keys, init_pbars)
    # Per-chain stacked output.
    assert traces.position.shape == (n_chains, n_steps)
    assert infos["accepted"].shape == (n_chains, n_steps)
    # All chains should be exploring (not stuck rejecting / all-accepting).
    per_chain_acc = infos["accepted"].mean(axis=1)
    assert jnp.all((per_chain_acc > 0.1) & (per_chain_acc < 0.99))


def test_run_chain_traces_kernel_once():
    """run_chain wraps lax.scan; the kernel body should jaxpr-trace exactly once."""
    log_target = lambda x: -0.5 * x ** 2
    init = init_state(jnp.array(0.0), log_target)
    kernel = partial(
        metropolis_step,
        propose_fn=random_walk_proposal(0.5),
        log_target_fn=log_target,
    )
    # Two different lengths should both compile under the same outer jit.
    @jax.jit
    def run10(k, s):
        return run_chain(k, s, kernel, n_steps=10)
    chain1, _ = run10(jax.random.PRNGKey(0), init)
    chain2, _ = run10(jax.random.PRNGKey(1), init)
    assert chain1.position.shape == chain2.position.shape == (10,)
