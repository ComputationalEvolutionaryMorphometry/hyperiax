# Hyperiax: Tree traversals using JAX

## Introduction

Hyperiax is a framework for tree traversal and computations on large-scale tree. Its primary purpose is to facilitate efficient message passing and operation execution on large trees. Hyperiax uses [JAX](https://jax.readthedocs.io/en/latest/index.html) for fast execution and automatic differentiation. Hyperiax is developed and maintained by [CCEM, UCPH](https://www.ccem.dk/).

Initially, Hyperiax was designed specifically for phylogenetic analysis of biological shape data, particularly enabling statistical inference with continuous time stochastic processes along the edges of the trees. For this purpose, is integrated with [JAXGeometry](https://bitbucket.org/stefansommer/jaxgeometry/src/main/), a computational differential geometry toolbox implemented in JAX. However, Hyperiax's messaging system and operations are general, which means that they can be easily adapted for use in other contexts. With minor modifications, Hyperiax can be used for any application where fast tree-level computations are necessary. Included examples cover such cases with inference in Gaussian graphical models, phylogenetic mean computation, and recursive shape matching in binary trees.

## Installation
```bash
# Create a separate environment (Recommended)
conda create -n hyperiax python==3.12.1 -y
conda activate hyperiax

# Install Hyperiax directly using pip
pip install hyperiax

# Install Hyperiax from the repository, for the newest version
pip install git+https://github.com/ComputationalEvolutionaryMorphometry/hyperiax.git

# Install Hyperiax for development
git clone git@github.com:ComputationalEvolutionaryMorphometry/hyperiax.git
# or (if you haven't set up ssh)
git clone https://github.com/ComputationalEvolutionaryMorphometry/hyperiax.git
# and then install by
pip install -e hyperiax[dev]
# and optionally
pip install -e hyperiax[examples]
# to install the dependencies for all the example notebooks
```

## Code Examples
- Set up a tree
```python
# Initialize a tree with a height of 4 and a degree of 3
tree = hyperiax.tree.builders.symmetric_tree(h=4, degree=3)
# Visualize
tree.plot_tree()

# Initialize the data value in nodes and branch lengths with example initialized data
key = jax.random.PNGKey(0)
# Randomly initialized values and lengths
exmp_values = jax.random.normal(key, shape=(16, ))
exmp_lengths = jax.random.uniform(key, shape=(16, ))
# Assign the values and lengths by broadcasting
tree["value"] = init_values
tree["edge_length"] = init_lengths
# Initialize the noise within the tree
noisy_tree = hyperiax.tree.initializers.initialize_noise(tree, key, shape=(1, ))
```

- Define operations and executor
```python
# Define the function executed along the edge
@jax.jit
def down(noise, edge_length, parent_value, **kwargs):    # example down function, insert your own one
    return {"value": jnp.sqrt(edge_length) * noise + parent_value}

up = jaxtrees.models.functional.pass_up('value', 'edge_length')

@jax.jit
def fuse(child_value, child_edge_length, **kwargs):    # example fuse function, insert your own one
    child_edge_length_inv = 1. / child_edge_length
    res = jnp.einsum('b1,bd->d', child_edge_length_inv, child_value) / child_edge_length.sum()
    return {"value": res}
```
- Run the simulation
```python
# Wrap all the functions in one model
updown_model = hyperiax.model.lambdamodels.UpDownLambda(up, fuse, down)
# Define the executor and run it
exe = hyperiax.execution.DependencyTreeExecutor(updown_model, batch_size=5)
# Do the inference from bottom to top
inf_tree = exe.up(noisy_tree)
# Do the sampling from top to bottom
sample_tree = exe.down(noisy_tree)
```
See [Examples](https://github.com/ComputationalEvolutionaryMorphometry/hyperiax/wiki/Examples) for more specific examples.
## Documentation
- __Getting Started__: See [Getting-Started](https://github.com/ComputationalEvolutionaryMorphometry/hyperiax/wiki/Getting-Started)
- __Guidance__: See [Wiki](https://github.com/ComputationalEvolutionaryMorphometry/hyperiax/wiki)
- __Full API Documentation__: See Hyperiax API

## Todo

## Contribution
Contributions, issues and feature requests are all welcome! Please refer to the [contributing guidelines](./CONTRIBUTION.md) before you want to contribute to the project.

## Contact
If you have questions, please contact: 
