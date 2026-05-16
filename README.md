# Hyperiax: Tree traversals using JAX
<p align="center">
 <img width="300", height="250" src="./docs/figures/hyperiax_logo.png">
</p>

> **Note (May 2026):** the `v3` branch is a ground-up rewrite of the core API.
> The published `pip install hyperiax` still points at the legacy 0.x release;
> v3.0.0 is the upcoming first release of the new line.

## Introduction

_Hyperiax_ is a framework for tree traversal and computations on large-scale trees. Its primary purpose is to facilitate **efficient message passing** and **operation execution** on large trees. Hyperiax uses [JAX](https://jax.readthedocs.io/en/latest/index.html) for fast execution and automatic differentiation. _Hyperiax_ is currently developed and maintained by [CCEM, UCPH](https://www.ccem.dk/).

Initially, _Hyperiax_ was designed for phylogenetic analysis of biological shape data, particularly statistical inference with continuous-time stochastic processes along tree edges. Its messaging primitives are general — applicable to Gaussian graphical models, phylogenetic mean computation, recursive shape matching, and similar tree-structured problems.

## Installation (development, v3 branch)

Hyperiax v3 uses [`uv`](https://docs.astral.sh/uv/) to manage the Python environment.

```bash
# Install uv (macOS)
brew install uv

# Clone and enter the repo
git clone git@github.com:ComputationalEvolutionaryMorphometry/hyperiax.git
cd hyperiax
git switch v3

# Create the project venv with the dev group (test + lint + docs tooling).
uv sync --group dev

# Optional extras:
#   - io:        ete3-backed Newick I/O
#   - prebuilt:  diffrax + jax-tqdm for the BFFG / MCMC prebuilt models
#   - notebook:  jupyter + matplotlib + optax for the tutorial notebooks
uv sync --group dev --extra io --extra prebuilt --extra notebook

# Run tests
uv run pytest

# Enable pre-commit hooks (ruff lint + format, whitespace, yaml/toml checks)
uv run pre-commit install
```

Python 3.11 or newer is required.

## Project layout (v3)

```
hyperiax/
├── core/                 # L1 — Topology, Tree, sweep decorators (no external deps)
├── io/                   # L2 — Newick I/O via ete3
└── prebuilt/             # L2 — phylo_mean, BFFG (Gaussian/SDE), MCMC, SDE utilities
```

## Contribution

Contributions, issues and feature requests are welcome — please open an issue or PR on GitHub.

## Contact

For questions related to the Hyperiax project, please contact [Stefan Sommer](mailto:sommer@di.ku.dk).
