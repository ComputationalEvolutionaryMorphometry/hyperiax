# Hyperiax: Tree traversals using JAX
<p align="center">
 <img width="300", height="250" src="./docs/figures/hyperiax_logo.png">
</p>


## Introduction

Hyperiax is a framework for tree traversal and computations on large-scale tree. Its primary purpose is to facilitate efficient message passing and operation execution on large trees. Hyperiax uses [JAX](https://jax.readthedocs.io/en/latest/index.html) for fast execution and automatic differentiation. Hyperiax is developed and maintained by [CCEM, UCPH](https://www.ccem.dk/).

Initially, Hyperiax was designed specifically for phylogenetic analysis of biological shape data, particularly enabling statistical inference with continuous time stochastic processes along the edges of the trees. For this purpose, is integrated with [JAXGeometry](https://bitbucket.org/stefansommer/jaxgeometry/src/main/), a computational differential geometry toolbox implemented in JAX. However, Hyperiax's messaging system and operations are general, which means that they can be easily adapted for use in other contexts. With minor modifications, Hyperiax can be used for any application where fast tree-level computations are necessary. Included examples cover such cases with inference in Gaussian graphical models, phylogenetic mean computation, and recursive shape matching in binary trees.

## Installation
```bash
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
topology = symmetric_topology(height=3, degree=2)
# Visualize
plot_tree_text(topology)

# Create tree for fast computation
tree = HypTree(topology)

tree.add_property('noise', shape=(2,))
tree.add_property('edge_length', shape=(1,))
tree.add_property('value', shape=(2,))

# Initialize the data
key = jax.random.PNGKey(0)
# Randomly initialized values and lengths
tree.data['noise'] = jax.random.normal(key, shape=tree.data['noise'].shape)
```

- Define operations and executor
```python
# Define the function executed along the edge
@jax.jit
def down(noise, edge_length, parent_value, **kwargs):    # example down function, insert your own one
    return {"value": jnp.sqrt(edge_length) * noise + parent_value}

```
- Run the simulation
```python
# Wrap all the down pass in one model
downmodel = DownLambda(down_fn=down)
# Define the executor and run it
exe = OrderedExecutor(downmodel)
# Do the sampling from top to bottom
exe.down(noisy_tree)

# Find the simulated data via
# tree.iter_topology()
# or tree.data
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
If you experience problems or have technical questions, please open an issue. For questions related to the Hyperiax project or CCEM, please contact [Stefan Sommer](mailto:sommer@di.ku.dk).
