hyperiax — JAX tree traversals
==============================

**hyperiax** is a small, pure-JAX library for message passing on phylogenetic
/ rooted trees. Trees are immutable JAX pytrees; sweeps are decorator-style
``Tree -> Tree`` transforms that compose cleanly under ``@jax.jit`` and
``jax.lax.scan``.

.. code-block:: python

   import jax, jax.numpy as jnp
   import hyperiax as hx

   topo = hx.symmetric_topology(depth=3, degree=2)
   tree = hx.Tree.empty(topo, {"value": (2,)})
   tree = tree.at[topo.is_leaf].set(value=jnp.ones((8, 2)))

   @hx.up(reads_children=("value",), writes=("value",))
   def avg(node, children, params):
       return {"value": children.value.mean(0)}

   root_value = avg(tree)["value"][0]

What's in the box
-----------------

* **core**: :class:`~hyperiax.Topology`, :class:`~hyperiax.Tree`,
  :class:`~hyperiax.Schema`, the ``@up`` / ``@down`` sweep decorators, and
  the equal- / unequal-degree dispatch engine.
* **prebuilt**: BFFG for Gaussian and SDE transitions
  (van der Meulen & Sommer 2025), Metropolis-Hastings on arbitrary pytrees,
  LDDMM landmark dynamics, weighted phylo-mean.
* **io**: Newick read/write via ``ete3`` (optional extra).

Installation
------------

.. code-block:: bash

   pip install hyperiax                  # core only (jax + numpy)
   pip install 'hyperiax[io]'            # + Newick I/O
   pip install 'hyperiax[prebuilt-bffg]' # + diffrax (ODE BFFG path)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   notebooks/index

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index


Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
