"""Topology builders.

Constructors that produce a fully-derived :class:`Topology` from a compact
description. All builders bottom out in :meth:`Topology.from_parents`.
"""

from __future__ import annotations

import numpy as np

from .topology import Topology


def symmetric_topology(depth: int, degree: int) -> Topology:
    """A regular tree where every internal node has exactly ``degree`` children.

    A tree of ``depth=0`` contains just the root; ``depth=1`` has the root
    plus one level of ``degree`` leaves; and so on. Total node count is
    :math:`\\sum_{k=0}^{h} d^k = (d^{h+1} - 1) / (d - 1)` for ``d > 1``, and
    ``h + 1`` for ``d == 1``.
    """
    if depth < 0:
        raise ValueError(f"depth must be non-negative, got {depth}")
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")

    if degree == 1:
        size = depth + 1
        # Chain: parents[i] = i - 1, root self-parents.
        parents = np.arange(-1, depth, dtype=np.int32)
        parents[0] = 0
    else:
        size = (degree ** (depth + 1) - 1) // (degree - 1)
        parents = np.zeros(size, dtype=np.int32)
        prev_level_start = 0
        cursor = 1
        for level in range(1, depth + 1):
            n_at_level = degree ** level
            local = np.arange(n_at_level, dtype=np.int32)
            parents[cursor : cursor + n_at_level] = prev_level_start + local // degree
            prev_level_start = cursor
            cursor += n_at_level

    return Topology.from_parents(parents)


def from_parents(parents, *, names=None) -> Topology:
    """Thin re-export of :meth:`Topology.from_parents`."""
    return Topology.from_parents(parents, names=names)
