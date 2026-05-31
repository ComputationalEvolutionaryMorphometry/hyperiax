# Hyperiax

<p align="center">
  <img width="300" height="250" src="./docs/figures/hyperiax_logo.png" alt="Hyperiax logo">
</p>

Hyperiax is a pure-JAX library for message passing on rooted trees. It provides
immutable tree data structures, typed per-node arrays, and decorator-based
sweeps that compose with `jax.jit`, `jax.vmap`, and `jax.lax.scan`.

Hyperiax is designed for phylogenetic and tree-structured scientific computing,
including Gaussian graphical models, phylogenetic means, and guided inference
for diffusion processes on trees.

## Installation

Hyperiax requires Python 3.11 or newer.

```bash
pip install hyperiax
```

The core package depends only on JAX, jaxlib, and NumPy. For accelerator-backed
JAX installations, follow the official JAX installation instructions for your
platform before or after installing Hyperiax.

## Quick Start

```python
import jax.numpy as jnp
import hyperiax as hx

topology = hx.symmetric_topology(depth=2, degree=2)
tree = hx.Tree.empty(topology, {"value": (2,)})

leaf_count = int(topology.is_leaf.sum())
tree = tree.at[topology.is_leaf].set(value=jnp.ones((leaf_count, 2)))


@hx.up(reads_children=("value",), writes=("value",))
def average_children(node, children, params):
    return {"value": children.value.mean(0)}


result = average_children(tree)
root_value = result.value[0]
```

Sweeps are ordinary `Tree -> Tree` functions. The `@hx.up` and `@hx.down`
decorators declare which fields are read and written, so dispatch remains
explicit and JAX-friendly.

## Architecture

```text
hyperiax/
├── core/      # Topology, Tree, Schema, views, sweep dispatch, builders
├── utils/     # Pure-JAX ODE and SDE solvers
└── prebuilt/  # Ready-to-use sweeps for BFFG and phylogenetic means
```

### Core

`hyperiax.core` contains the public primitives:

- `Topology` for rooted tree structure.
- `Tree` for immutable per-node JAX arrays.
- `Schema` for field shape and dtype validation.
- `@hx.up` and `@hx.down` for message-passing sweeps.
- Newick helpers for importing and exporting rooted trees.

### Utilities

`hyperiax.utils` contains pure-JAX numerical helpers, including ODE and SDE
solvers used by the prebuilt guided-inference routines.

### Prebuilt Sweeps

`hyperiax.prebuilt` contains focused implementations for common tree workflows:

- `phylo_mean` for weighted phylogenetic means.
- `bffg` for Backward Filtering Forward Guiding on discrete Gaussian and
  continuous SDE edges.

## Documentation

Tutorials and API reference are available in the project documentation:

- Quickstart and sweep-writing tutorials under `docs/source/notebooks/`.
- Public API reference under `docs/source/api/`.

## Reference

The BFFG implementation follows:

> van der Meulen, F. H. & Sommer, S. (2025). Backward Filtering Forward
> Guiding. JMLR 26(281), 1-51. https://arxiv.org/abs/2505.18239

## Contact

Hyperiax is maintained by CCEM, University of Copenhagen. For technical
questions, please open a GitHub issue or contact
[Stefan Sommer](mailto:sommer@di.ku.dk).
