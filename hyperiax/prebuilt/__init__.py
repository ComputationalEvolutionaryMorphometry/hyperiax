"""hyperiax.prebuilt — recipes built on the core API.

This is an L2 submodule: each prebuilt is a small wrapper around
:mod:`hyperiax.core` that ships ready-made sweeps for common tree
inference tasks (phylogenetic mean estimation, BFFG filters).
Importing :mod:`hyperiax.prebuilt` itself does not pull in any optional
dependencies; individual prebuilts may, but they import lazily.

The BFFG public API is still in flux (the ``discrete_*`` / ``continuous_*``
rename is WIP), so the BFFG sweeps are not re-exported here — pull them
directly from :mod:`hyperiax.prebuilt.bffg` (e.g. ``from
hyperiax.prebuilt.bffg import discrete_bf_sweep``) until the surface
stabilises.

For Markov-chain Monte Carlo, use NumPyro: the BFFG-guided forward map
composes cleanly with ``numpyro.factor`` inside a NumPyro ``model``, and
the built-in samplers (``NUTS``, ``HMC``) or any custom ``MCMCKernel``
drive the chain. See ``docs/source/notebooks/05_gaussian_bffg.ipynb`` and
``06_gaussian_nuts.ipynb`` for worked examples.
"""

from . import bffg
from .phylo_mean import phylo_mean

__all__ = [
    "bffg",
    "phylo_mean",
]
