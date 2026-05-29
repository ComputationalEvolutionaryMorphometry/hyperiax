"""SDE solvers: Euler-Maruyama (strong 0.5) and Milstein (strong 1.0).

Pure-JAX, tree-agnostic numerics — part of the L1 ``hyperiax.utils``
layer (imports only ``jax`` / ``numpy`` / stdlib). Solvers are frozen,
hashable dataclasses exposing a single ``step``; :func:`solve_sde` drives
them over a time grid with ``jax.lax.scan`` so the whole integration
composes under ``jit`` / ``vmap`` / ``scan``.

Conventions (mirroring standard SDE-solver libraries):

- ``drift(t, y, args) -> (state,)`` is the deterministic term.
- ``diffusion(t, y, args) -> (state, noise)`` is the matrix ``G``; the
  stochastic term is ``G @ dW``.
- ``ts`` are cumulative time points (shape ``(n_steps + 1,)``); ``dWs``
  are the Brownian increments (shape ``(n_steps, noise)``).

:class:`Milstein` assumes **commutative noise**, so the iterated Itô
integrals collapse to ``½(dW_i dW_j − δ_ij dt)`` and no Lévy areas are
needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


# ── Kronecker-factored linear algebra (landmark-style state) ───────
def dot(A: jax.Array, v: jax.Array) -> jax.Array:
    """Compute ``(A ⊗ I_d) @ v`` on a flattened state vector.

    ``v`` is interpreted as shape ``(n, d)`` where ``n = A.shape[0]``;
    the result is flattened back to ``(n·d,)``.
    """
    return jnp.einsum("ij,jd->id", A, v.reshape((A.shape[0], -1))).flatten()


def solve(A: jax.Array, v: jax.Array) -> jax.Array:
    """Solve ``(A ⊗ I_d) x = v`` for ``x``, flattened (``dot`` with ``A⁻¹``)."""
    return jnp.linalg.solve(A, v.reshape((A.shape[0], -1))).flatten()


def dts(T: float = 1.0, n_steps: int = 100) -> jax.Array:
    """Uniform time increments over ``[0, T]`` — shape ``(n_steps,)``.

    Cumulative time points for :func:`solve_sde` / :func:`solve_ode` are
    then ``jnp.concatenate([jnp.zeros(1), jnp.cumsum(dts(...))])``.
    """
    return jnp.full((n_steps,), T / n_steps)


# ── solvers ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EulerMaruyama:
    """Euler-Maruyama scheme. Strong order 0.5, weak order 1.0."""

    def step(self, drift, diffusion, t, dt, y, dW, args):
        return y + drift(t, y, args) * dt + diffusion(t, y, args) @ dW


@dataclass(frozen=True)
class Milstein:
    """Milstein scheme for commutative noise. Strong order 1.0.

    Adds the correction ``½ Σ_{j,k} (G·∇)G · (dW_j dW_k − δ_jk dt)``; the
    diffusion Jacobian ``∂G/∂y`` is obtained with :func:`jax.jacfwd`.
    """

    def step(self, drift, diffusion, t, dt, y, dW, args):
        G = diffusion(t, y, args)  # (state, noise)
        J = jax.jacfwd(lambda yy: diffusion(t, yy, args))(y)  # (state, noise, state)
        # M[j,k] = dW_j dW_k − δ_jk dt
        M = jnp.outer(dW, dW) - dt * jnp.eye(dW.shape[0])
        # mil_i = ½ Σ_{j,k,m} G[m,j] J[i,k,m] M[j,k]
        mil = 0.5 * jnp.einsum("mj,ikm,jk->i", G, J, M)
        return y + drift(t, y, args) * dt + G @ dW + mil


# ── driver ─────────────────────────────────────────────────────────
def solve_sde(drift, diffusion, y0, ts, dWs, *, solver=EulerMaruyama(), args=None):
    """Integrate ``dy = drift(t,y) dt + diffusion(t,y) dW`` over the grid ``ts``.

    Args:
        drift: ``(t, y, args) -> (state,)``.
        diffusion: ``(t, y, args) -> (state, noise)``.
        y0: initial state ``(state,)``.
        ts: cumulative time points ``(n_steps + 1,)``.
        dWs: Brownian increments ``(n_steps, noise)``.
        solver: a solver instance (default :class:`EulerMaruyama`).
        args: passed through to ``drift`` / ``diffusion``.

    Returns:
        Trajectory ``ys`` of shape ``(n_steps + 1, state)`` including ``y0``.
    """
    step_dts = jnp.diff(ts)

    def body(carry, inp):
        t, y = carry
        dt, dW = inp
        y_next = solver.step(drift, diffusion, t, dt, y, dW, args)
        return (t + dt, y_next), y

    (_, y_last), ys = jax.lax.scan(body, (ts[0], y0), (step_dts, dWs))
    return jnp.concatenate([ys, y_last[None]])
