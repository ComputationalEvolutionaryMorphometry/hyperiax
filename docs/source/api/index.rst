API reference
=============

Everything in :mod:`hyperiax` lives behind a small public surface; the
sections below list it module by module. Each entry links to a per-symbol
page generated from its docstring.

hyperiax.core
-------------

Tree topology, immutable Tree pytree, schema, views, sweep decorators.

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   hyperiax.Topology
   hyperiax.Tree
   hyperiax.Schema
   hyperiax.FieldSpec
   hyperiax.SweepFn
   hyperiax.Node
   hyperiax.Children
   hyperiax.ChildrenAxis

.. autosummary::
   :toctree: generated

   hyperiax.up
   hyperiax.down
   hyperiax.symmetric_topology
   hyperiax.from_parents
   hyperiax.from_newick
   hyperiax.to_newick
   hyperiax.HyperiaxError
   hyperiax.MissingField
   hyperiax.SchemaMismatch

hyperiax.prebuilt
-----------------

Ready-to-use sweeps and helpers for common message-passing tasks.

.. rubric:: Phylogenetic weighted mean

.. autosummary::
   :toctree: generated

   hyperiax.prebuilt.phylo_mean

.. rubric:: BFFG — discrete-edge (linear-Gaussian) sweeps

.. autosummary::
   :toctree: generated

   hyperiax.prebuilt.bffg.discrete_schema
   hyperiax.prebuilt.bffg.init_discrete_tree
   hyperiax.prebuilt.bffg.discrete_bf_sweep
   hyperiax.prebuilt.bffg.discrete_forward_sweep
   hyperiax.prebuilt.bffg.discrete_fg_sweep

.. rubric:: BFFG — continuous-edge (SDE) sweeps

.. autosummary::
   :toctree: generated

   hyperiax.prebuilt.bffg.continuous_schema
   hyperiax.prebuilt.bffg.init_continuous_tree
   hyperiax.prebuilt.bffg.continuous_bf_sweep
   hyperiax.prebuilt.bffg.continuous_forward_sweep
   hyperiax.prebuilt.bffg.continuous_fg_sweep

For MCMC over BFFG-guided latents and hyperparameters, hyperiax composes
with `NumPyro <https://num.pyro.ai/>`_ — see :doc:`../notebooks/05_gaussian_bffg`
and :doc:`../notebooks/06_gaussian_nuts` for worked examples.
