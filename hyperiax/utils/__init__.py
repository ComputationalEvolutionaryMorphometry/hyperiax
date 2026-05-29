"""hyperiax.utils — pure-JAX numerical utilities (L1).

Standalone numerics with no dependency on the tree primitives in
:mod:`hyperiax.core`: ODE and SDE solvers usable as the integration
backbone of :mod:`hyperiax.prebuilt` sweeps. Imports only ``jax`` /
``numpy`` / stdlib. The Kronecker helpers ``dot`` / ``solve`` and the
``dts`` grid helper live in :mod:`hyperiax.utils.sde`.
"""

from .ode import RK4, Euler, Heun, solve_ode
from .sde import EulerMaruyama, Milstein, solve_sde

__all__ = [
    "RK4",
    "Euler",
    "EulerMaruyama",
    "Heun",
    "Milstein",
    "solve_ode",
    "solve_sde",
]
