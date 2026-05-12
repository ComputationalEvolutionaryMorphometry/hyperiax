"""hyperiax.prebuilt — recipes built on the core API.

This is an L2 submodule: each prebuilt is a small wrapper around
:mod:`hyperiax.core` that ships ready-made sweeps for common tree
inference tasks (phylogenetic mean estimation, BFFG filters, etc.).
Importing :mod:`hyperiax.prebuilt` itself does not pull in any optional
dependencies; individual prebuilts may, but they import lazily.
"""

from . import bffg_gaussian, lddmm, sde, shape_kernels
from .bffg_gaussian import (
    gaussian_down_unconditional,
    gaussian_up,
    init_gaussian_leaves,
)
from .lddmm import lddmm_covariance, lddmm_drift
from .phylo_mean import phylo_mean

__all__ = [
    "bffg_gaussian",
    "gaussian_down_unconditional",
    "gaussian_up",
    "init_gaussian_leaves",
    "lddmm",
    "lddmm_covariance",
    "lddmm_drift",
    "phylo_mean",
    "sde",
    "shape_kernels",
]
