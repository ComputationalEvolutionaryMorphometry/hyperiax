"""ODE solvers: fixed-step explicit Runge-Kutta (Euler, Heun, RK4).

Pure-JAX, tree-agnostic — part of the L1 ``hyperiax.utils`` layer
(imports only ``jax`` / ``numpy`` / stdlib). Fixed-step (no adaptive
control): the step count is static, so the integrator composes cleanly
under ``jit`` / ``vmap`` / ``scan`` and the solution lands exactly on the
requested grid ``ts`` — no interpolation needed.

``vector_field(t, y, args) -> dy/dt``. Solvers are frozen, hashable
dataclasses exposing a single ``step``; :func:`solve_ode` drives them
over the grid with ``jax.lax.scan``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Euler:
    """Explicit (forward) Euler. Order 1."""

    def step(self, f, t, dt, y, args):
        return y + dt * f(t, y, args)


@dataclass(frozen=True)
class Heun:
    """Heun's method (explicit trapezoidal / RK2). Order 2."""

    def step(self, f, t, dt, y, args):
        k1 = f(t, y, args)
        k2 = f(t + dt, y + dt * k1, args)
        return y + 0.5 * dt * (k1 + k2)


@dataclass(frozen=True)
class RK4:
    """Classic 4-stage Runge-Kutta. Order 4."""

    def step(self, f, t, dt, y, args):
        k1 = f(t, y, args)
        k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1, args)
        k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2, args)
        k4 = f(t + dt, y + dt * k3, args)
        return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_ode(vector_field, y0, ts, *, solver=RK4(), args=None):
    """Integrate ``dy/dt = vector_field(t, y, args)`` over the grid ``ts``.

    Args:
        vector_field: ``(t, y, args) -> dy/dt``.
        y0: initial state, the value at ``ts[0]``.
        ts: time points ``(n_steps + 1,)`` (need not be uniform).
        solver: a solver instance (default :class:`RK4`).
        args: passed through to ``vector_field``.

    Returns:
        ``ys`` of shape ``(n_steps + 1, *y0.shape)`` with ``ys[0] == y0``.
    """
    step_dts = jnp.diff(ts)

    def body(carry, dt):
        t, y = carry
        y_next = solver.step(vector_field, t, dt, y, args)
        return (t + dt, y_next), y

    (_, y_last), ys = jax.lax.scan(body, (ts[0], y0), step_dts)
    return jnp.concatenate([ys, y_last[None]])
