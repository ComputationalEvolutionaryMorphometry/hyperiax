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

The body expresses its per-child work via :meth:`Children.map`, so it
runs unchanged on both equal- and unequal-degree topologies.
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

    def _per_child(c):
        edge = c.edge_length      # () typically (scalar per child)
        val = c.estimated_value   # (*value_trailing)
        # Broadcast edge to match the value's trailing rank so the fields
        # produced here share trailing shape and combine cleanly after fusion.
        edge_b = edge.reshape(edge.shape + (1,) * (val.ndim - edge.ndim))
        return {"weighted": val / edge_b, "inv_edge": 1.0 / edge_b}

    @up(
        reads_children=("estimated_value", "edge_length"),
        writes=("estimated_value",),
    )
    def _sweep(node, children, params):
        msgs = children.map(_per_child)
        return {"estimated_value": msgs.weighted.sum(0) / msgs.inv_edge.sum(0)}

    return _sweep
