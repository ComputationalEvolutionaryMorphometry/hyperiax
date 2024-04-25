=====
Usage
=====

.. _installation:

------------
Installation
------------

.. code-block:: bash

    # Install Hyperiax directly using pip
    $ pip install hyperiax

    # Install Hyperiax from the repository, for the newest version
    $ pip install git+https://github.com/ComputationalEvolutionaryMorphometry/hyperiax.git

    # Install Hyperiax for development
    $ git clone git@github.com:ComputationalEvolutionaryMorphometry/hyperiax.git
    # or (if you haven't set up ssh)
    $ git clone https://github.com/ComputationalEvolutionaryMorphometry/hyperiax.git
    # and then install by
    $ pip install -e hyperiax[dev]
    # and optionally
    $ pip install -e hyperiax[examples]
    # to install the dependencies for all the example notebooks

------------
Build a tree
------------

Like many tree implementations, Hyperiax's basic unit is called a ``TreeNode``, which contains four attributes: ``parent``, ``data``, ``children``, and ``name``: ``parent`` records the parent node, which is also a ``TreeNode``; ``data`` is implemented as a dictionary, which supports storing different types of data identified by their keys; ``children`` list contains the child nodes; and ``name`` is used to characterize different nodes.

.. code-block:: python
    @dataclass
    class TreeNode:
        parent: TreeNode = None
        data: Dict = field(default_factory=dict)
        children: List[TreeNode] = None
        name: str = None

``TreeNode`` supports direct access and manipulations of the data, for example

>>> from hyperiax.tree.tree import TreeNode
>>> node = TreeNode(data = {"value": 0.1})
>>> print(node["value"])
0.1
>>> node["noise"] = 0.5
>>> print(node["noise"])
0.5
>>> del node["noise"]
>>> print(node["noise"])
KeyError: 'noise'

If you want to create your own ``Node ``class with extra properties, consider inheriting ``TreeNode`` and passing your ``Node`` to the ``builders``.

The tree in Hyperiax is realized in ``HypTree``, which also supports direct access to the data stored in the tree through keys. To fastly build a ``HypTree`` object, Hyperiax provides three different ways:

:py:func:`symmetric_tree()`         Build a symmetric tree with a given height and a fixed degree for all the nodes
:py:func:`asymmetric_tree()`        Build an asymmetric binary tree with a given height
:py:func:`tree_from_newick()`       Build an arbitrary tree from Newick strings

^^^^^^^^^^^^^^
Symmetric tree
^^^^^^^^^^^^^^
:py:func:`symmetric_tree()` is used to create a symmetric tree, which takes two main arguments ``h`` and ``degree``. A customized ``Node`` class can be passed through the argument ``new_node`` (default is ``TreeNode``). To create a tree with a height of 3 and a degree of 2:

>>> from hyperiax.tree.builders import symmetric_tree, asymmetric_tree, tree_from_newick
>>> tree = symmetric_tree(h=3, degree=2)
>>> tree.plot_tree_text()
       *
   ┌───┴───┐
   *       *
 ┌─┴─┐   ┌─┴─┐
 *   *   *   *
┌┴┐ ┌┴┐ ┌┴┐ ┌┴┐
* * * * * * * *

^^^^^^^^^^^^^^^^^^^^^^
Asymmetric binary tree
^^^^^^^^^^^^^^^^^^^^^^
If you want to create an asymmetric binary tree with a height of ``h``, use :py:func:`asymmetric_tree()`:

>>> tree = asymmetric_tree(h=3)
>>> tree.plot_tree_text()
    *
  ┌─┴─┐
  *   *
 ┌┴─┐
 *  *
┌┴┐
* *

^^^^^^^^^^^
Newick tree
^^^^^^^^^^^
The most flexible way of creating a tree is to use the `Newick format <https://en.wikipedia.org/wiki/Newick_format>`, which is also the format in which Hyperiax stores a defined tree structure. Hyperiax supports different types of Newick formats. To create a tree, you need a Newick string:

>>> newick_str = "((,),);"
>>> tree = tree_from_newick(newick_str)
>>> tree.plot_tree_text()
  *
 ┌┴─┐
 *  *
┌┴┐
* *
>>> newick_str = "((A:0.2,B:0.1)C:0.3,D:0.1)F:0.5;"
>>> tree = tree_from_newick(newick_str)
>>> tree.plot_tree_text()
  F
 ┌┴─┐
 C  D
┌┴┐
A B

^^^^^^^^^^^^^^^^^^^
Initialize the tree
^^^^^^^^^^^^^^^^^^^
After determining the topology of the tree, you would probably like to assign data at each node and specify the edge length (which is unnecessary if you use a Newick string to create the tree).

If you have a list or array to store concrete node data, you can assign them to each node by implicit broadcasting and access them afterward through the same key access as the TreeNode:

>>> import jax.numpy as jnp
>>> exmp_value = jnp.array([1.0, 2.0])      # Example data values to be assigned, the value stored in each node has a shape of (2, )
>>> tree["value"] = exmp_value
>>> for i, val in enumerate(tree["value"]): # Access the values by iterating the tree
>>>     print(f"Node {i} with value={val}", end=' ')
Node 0 with value=[1. 2.] Node 1 with value=[1. 2.] Node 2 with value=[1. 2.] Node 3 with value=[1. 2.] Node 4 with value=[1. 2.]

Or if you don't know the specific data but know the distribution, e.g. :math:`\mathcal{N}(0,I)`, you may use :py:func:`initialize_noise()` to initialize all the nodes with random samples with one call:

>>> import jax
>>> from hyperiax.tree.initializers import initialize_noise, initialize_noise_leaves
>>> key = jax.random.PRNGKey(0)                         # pesudo random generator key required by JAX
>>> tree = initialize_noise(tree, key, (2, ))           # initialize normally distributed noise with the shape of (2,)
>>> tree = initialize_noise_leaves(tree, key, (2, ))    # initialize the noise only on leaves

-------------------------------------------
Create your tree functions and execute them
-------------------------------------------

One of Hyperiax's nice features is that it allows you to execute your functions defined between nodes fast and parallel. Two scenarios might occur:
* Executing some functions through the whole tree, e.g., computing the mean root based on the leaves.
* Updating tree parameters locally, e.g., MCMC parameter update for a certain node.

Hyperiax provides two basic executors for these two purposes, ``OrderedExecutor`` (for the whole tree execution) and ``UnorderedExecutor`` (for local execution), respectively, together with some concrete executors inherited from these two. Let's take a look at them one by one.

^^^^^^^^^^^^^^^^^^^^^^^^
The whole tree execution
^^^^^^^^^^^^^^^^^^^^^^^^
In general, there are three catalogues of functions you can apply to the whole tree execution: ``up``, ``down``, and ``fuse``.

* ``down``: The ``down`` function is defined on a single edge :math:`(u,v)`, where :math:`u` is the source node and :math:`v` is the target node and is used to compute the new value of :math:`v` based on the current values of both :math:`u` and :math:`v`, with the weight depends on the edge length. In the following down function, each node contains ``noise``; after being the :math:`v` part of a down call, it also contains ``value``. This means we can always get the ``parent_value`` since the order of the down call flows downward in the tree. Notice that any values can be obtained from :math:`u` by prefixing the key by ``parent_``, values in :math:`v` are simply passed by their key.

.. code-block:: python
    import jax.numpy as jnp

    @jax.jit                                                                # JIT compilation for fast computation
    def down(noise, edge_length, parent_value, **kargs) -> dict:
        return {'value': jnp.sqrt(edge_length) * noise + parent_value}      # a simple computation, replace it by yours

* ``up``: The ``up`` function is to serve the ``fuse`` function. It acts as the messenger to collect the data in source nodes and pack them for the ``fuse`` operation. It communicates a dictionary of values to be passed to the fuse function along with the other child nodes. Let's say that we are interested in just passing the ``value`` and ``edge_lengths``.

.. code-block:: python
    @jax.jit
    def up(value, edge_length,**args):
        return {'value': value, 'edge_length': edge_length}

Since this notation can be a bit cumbersome, we do provide the shorthand :py:func:`pass_up()`, where you simply specify the keys to pass up. Instead, we could write:

.. code-block:: python
    up = jaxtrees.models.functional.pass_up('value', 'edge_length')

* ``fuse``: The ``fuse`` function is responsible for combining all of the messages from the child nodes passed by the ``up`` function into a single parent node.

.. code-block:: python
    def fuse(child_value,child_edge_length, **kwargs):                                  # example fuse function, replace it by yours

        childrent_inv = 1 / child_edge_lengthf

        result = jnp.einsum('c1,cd->d',childrent_inv, child_value)/childrent_inv.sum()  # weight the children nodes by their edge lengths
        return {'value': result}

In order to execute these functions, you need to use ``OrderedExecutor`` and its derived classes. So far, Hyperiax provides two different ordered executors: ``DependencyTreeExecutor`` and ``LevelwiseExecutor``. In most cases, ``DependencyTreeExecutor`` is preferred for better performance unless you require your function to act level-wise, where ``LevelwiseExecutor`` can be used. To actually use them, you need to wrap all your functions into a ``lambdamodel``, which gives a simple interface to the executor.

>>> from hyperiax.models.lambdamodels import UpdownLambda
>>> from hyperiax.execution.dependencyexecutor import DependencyTreeExecutor
>>> updown_model = UpDownLambda(up, fuse, down)                 # wrap the functions into a lambdamodel
>>> exe = DependencyTreeExecutor(updown_model, batch_size=4)    # define the executor with the amount of batched nodes as 4
>>> inf_tree = exe.up(tree)                                     # do the inference from bottom to top
>>> sample_tree = exe.down(tree)                                # do the sampling from top to bottom

^^^^^^^^^^^^^^^^^
Local tree update
^^^^^^^^^^^^^^^^^
In some cases, local updates may be needed instead of executing the function through the entire tree, like MCMC parameter sampling for certain nodes, where the ``OrderedExecutor`` is no longer available since the update depends on the neighbors. Instead, ``UnorderedExecutor`` is designed for this case. Compared with ``OrderedExecutor``, where a key method :py:func:`_determine_execution_order()` is used to determine the order of the whole tree, in ``UnorderedExecutor``, this is replaced by :py:func:`_determine_execution_pools()`, which stores the pending nodes that can be in any order. Any new unordered executor should inherit ``UnorderedExecutor`` with rewritten :py:func:`_determine_executor_pools()` and :py:func:`_iter_pools()` methods. Hyperiax implemented a classical unordered executor called ``RedBlackExecutor``, which treats the tree as a red-black tree and executes red and black parts alternatively. Besides the executor, the update function should inherit from the base class ``UpdateModel`` with the implemented :py:func:`update()` method.

Besides the executor, Hyperiax provides two different parameter types, ``FixedParameter`` and ``VarianceParameter``, to distinguish between the fixed parameters and variable parameters; the latter is usually assumed to follow a Gamma distribution. A ``VarianceParameter`` object has a :py:func:`propose()` method, which shall return a new ``VarianceParameter`` object with a new sampled value given the previous value.

.. code-block:: python
    from hyperiax.models.updatemodel import UpdateModel
    class MCMC(UpdateModel):        # define the MCMC updating function
        def update(self, parent_value,children_values,node_value, parameters):
            if not children_values:
                return {'noise': parent_value['noise']}, True
            if not parent_value:
                return {'noise': children_values['noise'].mean(0)}, True
            
            parent_noise = parent_value['noise']
            children_noise = children_values['noise'].mean(0)

            result = (1-parameters['lambd'])*parent_noise+parameters['lambd'] *children_noise

            return {'noise': result}, True

    from hyperiax.mcmc.parameterstore import ParameterStore
    from hyperiax.mcmc.fixedparameter import FixedParameter
    from hyperiax.mcmc.varianceParameter import VarianceParameter
    from hyperiax.execution.redblackexecutor import RedBlackExecutor
    params = ParameterStore({
        'lambd': FixedParameter(value=0.5),    # A fixed parameter "lambd" with a value of 0.5
        'alpha': VarianceParameter(value=2)    # A random parameter "alpha" with an initial value of 2, with a proposal variance of 0.01 by default
    })

>>> model = MCMC()                              # instantiate model and executor
>>> exe = RedBlackExecutor(model)
>>> it = exe.get_iterator(noise_tree)           # execute the updates
>>> for node in it:
>>>     key, subkey = jax.random.split(key)
>>>     proposed = params.propose(subkey)
>>>     accepted = exe.update(node, proposed.values())
>>>     if accepted:
>>>         params = proposed

-------------------------
Save the tree and load it
-------------------------
Finally, if you have a tree instance and would like to store the topology for further use, you can call the instance's :py:func:`tree_to_newick()` method. This method converts the tree object generated by any of the methods mentioned before into the Newick representation and easily stores it as a string. You can also load it using :py:func:`tree_from_newick()`.

>>> tree = symmetric_tree(h=3, degree=3)
>>> tree.plot_tree_text()
                          *
        ┌─────────────────┼─────────────────┐
        *                 *                 *
  ┌─────┼─────┐     ┌─────┼─────┐     ┌─────┼─────┐
  *     *     *     *     *     *     *     *     *
┌─┼─┐ ┌─┼─┐ ┌─┼─┐ ┌─┼─┐ ┌─┼─┐ ┌─┼─┐ ┌─┼─┐ ┌─┼─┐ ┌─┼─┐
* * * * * * * * * * * * * * * * * * * * * * * * * * *
>>> tree_newick = tree.tree_to_newick()
>>> print("Newick string:"+tree_newick)
Newick string:((,,),(,,),(,,));
>>> new_tree = tree_from_newick(tree_newick)
>>> new_tree.plot_tree_text()
                          *
        ┌─────────────────┼─────────────────┐
        *                 *                 *
  ┌─────┼─────┐     ┌─────┼─────┐     ┌─────┼─────┐
  *     *     *     *     *     *     *     *     *
┌─┼─┐ ┌─┼─┐ ┌─┼─┐ ┌─┼─┐ ┌─┼─┐ ┌─┼─┐ ┌─┼─┐ ┌─┼─┐ ┌─┼─┐
* * * * * * * * * * * * * * * * * * * * * * * * * * *