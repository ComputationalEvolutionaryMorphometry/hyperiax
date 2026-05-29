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

Python 3.11 or newer is required. Hyperiax v3 uses
[`uv`](https://docs.astral.sh/uv/) to manage the Python environment —
`pyproject.toml` is the single source of truth and the checked-in `uv.lock`
pins a reproducible resolution.

```bash
# Install uv (macOS)
brew install uv

# Clone and enter the repo
git clone git@github.com:ComputationalEvolutionaryMorphometry/hyperiax.git
cd hyperiax
git switch v3

# Create the project venv: core CPU runtime + dev group (test + lint + docs).
uv sync --group dev

# Run tests
uv run pytest

# Enable pre-commit hooks (ruff lint + format, whitespace, yaml/toml checks)
uv run pre-commit install
```

The core runtime is **CPU JAX** (`jax`, `jaxlib`, `numpy`) — the cleanest
install, works everywhere. For a CUDA 12 machine, add the single `gpu` extra,
which layers the CUDA jaxlib plugin on top:

```bash
uv sync --group dev --extra gpu
```

Plain `pip` works too, with `pyproject.toml` as the source of truth:

```bash
pip install -e .            # core (CPU)
pip install -e '.[gpu]'     # core + CUDA 12
```

## Project layout (v3)

```
hyperiax/
├── core/                 # L1 — Topology, Tree, sweep decorators (no external deps)
├── io/                   # L2 — Newick I/O
└── prebuilt/             # L2 — phylo_mean, BFFG (Gaussian/SDE), MCMC, SDE utilities
```

## Contribution

Contributions, issues and feature requests are welcome — please open an issue or PR on GitHub.

## Contact

For questions related to the Hyperiax project, please contact [Stefan Sommer](mailto:sommer@di.ku.dk).
