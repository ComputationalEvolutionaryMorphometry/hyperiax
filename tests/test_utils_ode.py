"""Fixed-step ODE solvers in hyperiax.utils.ode."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperiax.utils.ode import RK4, Euler, Heun, solve_ode


def test_solve_ode_returns_grid_with_initial_value():
    ts = jnp.linspace(0.0, 1.0, 11)
    ys = solve_ode(lambda t, y, a: y, jnp.array([1.0]), ts, solver=RK4())
    assert ys.shape == (11, 1)
    np.testing.assert_allclose(np.asarray(ys[0]), [1.0])


def test_rk4_matches_exponential_to_high_accuracy():
    """dy/dt = y, y(0)=1 -> e^t; RK4 should be ~float32-exact."""
    ts = jnp.linspace(0.0, 1.0, 51)
    ys = solve_ode(lambda t, y, a: y, jnp.array([1.0]), ts, solver=RK4())
    np.testing.assert_allclose(float(ys[-1, 0]), float(jnp.exp(1.0)), rtol=1e-5)


def test_rk4_solves_harmonic_oscillator():
    """y'' = -y as a first-order system: (x, v)' = (v, -x); x(0)=1,v(0)=0
    has exact solution x(t)=cos t, v(t)=-sin t."""
    ts = jnp.linspace(0.0, 2.0 * jnp.pi, 400)
    ys = solve_ode(
        lambda t, y, a: jnp.array([y[1], -y[0]]),
        jnp.array([1.0, 0.0]),
        ts,
        solver=RK4(),
    )
    np.testing.assert_allclose(float(ys[-1, 0]), 1.0, atol=1e-4)  # cos(2π)
    np.testing.assert_allclose(float(ys[-1, 1]), 0.0, atol=1e-4)  # -sin(2π)


def test_solver_order_euler_worse_than_heun_worse_than_rk4():
    """At a fixed (coarse) step size, terminal error must strictly improve
    with solver order on the smooth exponential test."""
    ts = jnp.linspace(0.0, 1.0, 9)  # coarse: 8 steps
    truth = float(jnp.exp(1.0))
    err = {}
    for name, solver in (("euler", Euler()), ("heun", Heun()), ("rk4", RK4())):
        ys = solve_ode(lambda t, y, a: y, jnp.array([1.0]), ts, solver=solver)
        err[name] = abs(float(ys[-1, 0]) - truth)
    assert err["euler"] > err["heun"] > err["rk4"]


def test_solve_ode_args_threaded_through():
    """`args` reaches the vector field (here a decay rate)."""
    ts = jnp.linspace(0.0, 1.0, 101)
    ys = solve_ode(lambda t, y, k: -k * y, jnp.array([1.0]), ts, solver=RK4(), args=2.0)
    np.testing.assert_allclose(float(ys[-1, 0]), float(jnp.exp(-2.0)), rtol=1e-5)


def test_solve_ode_jit_and_vmap_compose():
    ts = jnp.linspace(0.0, 1.0, 51)

    @jax.jit
    def run(y0):
        return solve_ode(lambda t, y, a: y, y0, ts, solver=RK4())

    out = run(jnp.array([1.0]))
    assert out.shape == (51, 1)

    batched = jax.vmap(run)(jnp.array([[1.0], [2.0], [3.0]]))
    assert batched.shape == (3, 51, 1)
    np.testing.assert_allclose(np.asarray(batched[:, 0, 0]), [1.0, 2.0, 3.0])


@pytest.mark.parametrize("solver", [Euler(), Heun(), RK4()])
def test_all_solvers_run_and_preserve_shape(solver):
    ts = jnp.linspace(0.0, 0.5, 6)
    ys = solve_ode(lambda t, y, a: -y, jnp.array([1.0, 2.0]), ts, solver=solver)
    assert ys.shape == (6, 2)
