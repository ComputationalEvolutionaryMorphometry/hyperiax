Welcome to Hyperiax's documentation!
===================================

**Hyperiax** is a framework for tree traversal and computations on large-scale tree. Its primary purpose is to facilitate efficient message passing and operation execution on large trees. Hyperiax uses `JAX <https://jax.readthedocs.io/en/latest/index.html>`_ for fast execution and automatic differentiation. Hyperiax is developed and maintained by `CCEM, UCPH <https://www.ccem.dk/>`_.

Initially, Hyperiax was designed specifically for phylogenetic analysis of biological shape data, particularly enabling statistical inference with continuous time stochastic processes along the edges of the trees. For this purpose, is integrated with `JAXGeometry <https://bitbucket.org/stefansommer/jaxgeometry/src/main/>`_, a computational differential geometry toolbox implemented in JAX. However, Hyperiax's *messaging system* and *operations* are general, which means that they can be easily adapted for use in other contexts. With minor modifications, Hyperiax can be used for any application where fast tree-level computations are necessary. Included examples cover such cases with inference in Gaussian graphical models, phylogenetic mean computation, and recursive shape matching in binary trees.

Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   hyperiax

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`