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
   hyperiax.HyperiaxError
   hyperiax.MissingField
   hyperiax.SchemaMismatch

hyperiax.io
-----------

Newick read/write via :mod:`ete3` (optional ``[io]`` extra).

.. autosummary::
   :toctree: generated

   hyperiax.io.newick.read
   hyperiax.io.newick.write

hyperiax.prebuilt
-----------------

Ready-to-use sweeps and helpers for common message-passing tasks.

.. rubric:: Phylogenetic weighted mean

.. autosummary::
   :toctree: generated

   hyperiax.prebuilt.phylo_mean

.. rubric:: BFFG — Gaussian transitions (closed form)

.. autosummary::
   :toctree: generated

   hyperiax.prebuilt.gaussian_up
   hyperiax.prebuilt.gaussian_down_conditional
   hyperiax.prebuilt.gaussian_down_unconditional
   hyperiax.prebuilt.init_gaussian_leaves

.. rubric:: BFFG — SDE transitions (closed form + diffrax-ODE)

.. autosummary::
   :toctree: generated

   hyperiax.prebuilt.sde_up
   hyperiax.prebuilt.sde_down_conditional
   hyperiax.prebuilt.sde_down_unconditional
   hyperiax.prebuilt.init_sde_leaves
   hyperiax.prebuilt.propagate_v_T_to_v_0

.. rubric:: SDE utilities (Euler-Maruyama)

.. autosummary::
   :toctree: generated

   hyperiax.prebuilt.sde.forward
   hyperiax.prebuilt.sde.dts

.. rubric:: MCMC on JAX pytrees

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   hyperiax.prebuilt.MHState

.. autosummary::
   :toctree: generated

   hyperiax.prebuilt.init_state
   hyperiax.prebuilt.metropolis_step
   hyperiax.prebuilt.run_chain
   hyperiax.prebuilt.random_walk_proposal
   hyperiax.prebuilt.crank_nicolson_proposal

