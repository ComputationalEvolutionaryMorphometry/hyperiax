"""Public API and documentation consistency checks."""

from __future__ import annotations

import tomllib
from pathlib import Path

import jax.numpy as jnp

from hyperiax import Topology, Tree
from hyperiax.prebuilt.bffg import continuous_refine_anchor, continuous_schema

ROOT = Path(__file__).resolve().parent.parent


def test_propagate_linearization_public_sweep_matches_continuous_refine_anchor():
    """The public named sweep should perform the v3 continuous anchor refinement."""
    from hyperiax.prebuilt.bffg import propagate_linearization

    topo = Topology.from_parents([0, 0, 0])
    tree = Tree.empty(topo, continuous_schema(d=1, n_steps=1))
    tree = tree.set(
        prec_v=jnp.array([[[1.0]], [[2.0]], [[4.0]]]),
        ptnl_v=jnp.array([[0.0], [6.0], [20.0]]),
        anchor=jnp.array([[9.0], [8.0], [7.0]]),
        anchor_pa=jnp.array([[1.0], [1.0], [1.0]]),
    )

    expected = continuous_refine_anchor()(tree)
    actual = propagate_linearization(tree)

    assert jnp.allclose(actual.anchor, expected.anchor)
    assert jnp.allclose(actual.anchor_pa, expected.anchor_pa)


def test_notebook_extra_contains_notebook_runtime_dependencies():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    assert "notebook" in extras
    deps = "\n".join(extras["notebook"])
    for package in ("matplotlib", "jupyter", "joblib", "optax", "numpyro"):
        assert package in deps


def test_docs_and_notebooks_do_not_reference_removed_v3_apis():
    removed_tokens = (
        "hyperiax.io",
        "gaussian_up",
        "gaussian_down_conditional",
        "gaussian_down_unconditional",
        "init_gaussian_leaves",
        "children.X is `(k, *trailing)`",
        "dense `(scope, k, *trailing)`",
        "two different backends",
        "Metropolis-Hastings on arbitrary pytrees",
        "LDDMM landmark dynamics",
    )

    docs = [
        *(ROOT / "docs" / "source").glob("*.rst"),
        *(ROOT / "docs" / "source" / "notebooks").glob("*.ipynb"),
    ]
    offenders: list[str] = []
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for token in removed_tokens:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)}: contains {token!r}")

    assert not offenders, "\n".join(offenders)
