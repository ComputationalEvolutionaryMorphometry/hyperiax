"""Phylogenetic mean: edge-length-weighted average up-sweep.

The classical estimator for inner-node states given leaf observations on
a phylogenetic tree::

    \\hat{x}_p = \\frac{\\sum_c v_c / l_c}{\\sum_c 1 / l_c}

where the sum is over a parent ``p``'s children and ``l_c`` is the
length of the edge from ``p`` to child ``c``. Children with shorter
edges contribute more — the same prior you get under a Brownian-motion
diffusion model along edges.

Usage::

    tree = hx.Tree.empty(topo, {"estimated_value": (d,), "edge_length": ()})
    tree = tree.set(edge_length=...)
    tree = tree.at[topo.is_leaf].set(estimated_value=observed_leaves)

    sweep = hyperiax.prebuilt.phylo_mean()
    inferred = sweep(tree)
    root_estimate = inferred["estimated_value"][0]

Limitation
----------
Stage 7 implements the equal-degree path only — the user expression
``children.estimated_value / children.edge_length`` relies on JAX
broadcasting, which the unequal-degree ``ChildrenAxis`` proxy doesn't
yet expose. A ragged-tree variant arrives once the proxy gains
elementwise arithmetic (or a ``children.gather()`` fallback).
"""

from __future__ import annotations

from ..core.sweep import SweepFn, up


def phylo_mean() -> SweepFn:
    """Return a SweepFn that fills each non-leaf node's ``estimated_value``
    with the edge-length-weighted average of its children's
    ``estimated_value``.

    The sweep reads ``estimated_value`` and ``edge_length`` from each
    child and writes back ``estimated_value`` on the parent. Leaves are
    untouched.
    """

    @up(
        reads_children=("estimated_value", "edge_length"),
        writes=("estimated_value",),
    )
    def _sweep(node, children, params):
        # Per-parent under jax.vmap:
        #   children.edge_length     : (k, *edge_trailing) — typically (k,)
        #   children.estimated_value : (k, *value_trailing)
        edges = children.edge_length
        values = children.estimated_value
        # Broadcast `edges` over the trailing dims of `values` for the
        # elementwise division. The common case is scalar edges, vector
        # values: (k,) → (k, 1, ...) to match (k, d, ...).
        extra = values.ndim - edges.ndim
        if extra > 0:
            edges_b = edges.reshape(edges.shape + (1,) * extra)
        else:
            edges_b = edges
        weighted = values / edges_b
        inv_edges = 1.0 / edges_b
        return {"estimated_value": weighted.sum(0) / inv_edges.sum(0)}

    return _sweep
