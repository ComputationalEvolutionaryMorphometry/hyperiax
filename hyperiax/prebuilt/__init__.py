"""hyperiax.prebuilt — recipes built on the core API.

This is an L2 submodule: each prebuilt is a small wrapper around
:mod:`hyperiax.core` that ships ready-made sweeps for common tree
inference tasks (phylogenetic mean estimation, BFFG filters, etc.).
Importing :mod:`hyperiax.prebuilt` itself does not pull in any optional
dependencies; individual prebuilts may, but they import lazily.
"""

from . import bffg, mcmc, sde
from .bffg import (
    gaussian_down_conditional,
    gaussian_down_unconditional,
    gaussian_up,
    init_gaussian_leaves,
    init_sde_leaves,
    propagate_v_T_to_v_0,
    sde_down_conditional,
    sde_down_unconditional,
    sde_up,
)
from .mcmc import (
    MHState,
    crank_nicolson_proposal,
    init_state,
    metropolis_step,
    random_walk_proposal,
    run_chain,
)
from .phylo_mean import phylo_mean

__all__ = [
    "MHState",
    "bffg",
    "crank_nicolson_proposal",
    "gaussian_down_conditional",
    "gaussian_down_unconditional",
    "gaussian_up",
    "init_gaussian_leaves",
    "init_sde_leaves",
    "init_state",
    "mcmc",
    "metropolis_step",
    "phylo_mean",
    "propagate_v_T_to_v_0",
    "random_walk_proposal",
    "run_chain",
    "sde",
    "sde_down_conditional",
    "sde_down_unconditional",
    "sde_up",
]
