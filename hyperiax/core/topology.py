"""Tree topology — static structure, hashable, JAX-pytree-aware.

A :class:`Topology` carries everything the dispatcher needs to know about
the shape of a tree (parent pointers, level boundaries, per-level segment
layouts, etc.) but no per-node data. Every field is a :mod:`numpy` array,
so the object is Python-hashable and rides through ``jax.jit`` as static
``aux_data``. The dispatcher promotes these to ``jnp`` arrays at its
boundary; under JIT they become traced constants.

Convention
----------
- Nodes are laid out in **BFS order**: ``parents[i] < i`` for ``i > 0``.
- The root is node ``0`` and is its own parent: ``parents[0] == 0``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import numpy as np


@dataclass(frozen=True, eq=False)
class Topology:
    """Immutable tree topology with precomputed dispatch layouts."""

    # ── raw ──
    parents: np.ndarray  # (N,) int32; parents[0] == 0

    # ── derived: shape ──
    size: int
    depth: int  # = len(level_starts) - 2; a depth-0 tree has only the root
    level_starts: np.ndarray  # (depth+2,) int32; level i is [level_starts[i], level_starts[i+1])
    node_depths: np.ndarray  # (N,) int32

    # ── derived: masks ──
    child_counts: np.ndarray  # (N,) int32
    is_root: np.ndarray  # (N,) bool
    is_leaf: np.ndarray  # (N,) bool
    is_inner: np.ndarray  # (N,) bool

    # ── degree info ──
    max_degree: int
    equal_degree: bool

    # ── segment-reduction layout (used by the up-sweep dispatcher) ──
    # For each node, its *local* segment id within its level. Used as
    # ``segment_ids`` for ``jax.ops.segment_*`` during up-sweeps.
    pbuckets: np.ndarray  # (N,) int32
    # For each level, the parent node ids (in level-1) that the segments at
    # this level reduce into. ``pbuckets_ref[L]`` has shape ``(n_parents_at_L-1,)``.
    pbuckets_ref: tuple  # tuple[np.ndarray, ...]

    # ── informational ──
    names: tuple | None  # node names (e.g. from Newick); not used by dispatch

    # ── lifecycle ─────────────────────────────────────────────────────
    def __post_init__(self):
        # Frozen dataclass; bypass the immutability guard to cache the hash.
        object.__setattr__(self, "_struct_hash", hash(self.parents.tobytes()))

    # ── identity ──────────────────────────────────────────────────────
    def __hash__(self) -> int:  # type: ignore[override]
        return self._struct_hash  # type: ignore[attr-defined]

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, Topology):
            return False
        return (
            self.parents.shape == other.parents.shape
            and self._struct_hash == other._struct_hash  # type: ignore[attr-defined]
            and np.array_equal(self.parents, other.parents)
        )

    def __repr__(self) -> str:
        return (
            f"Topology(size={self.size}, depth={self.depth}, "
            f"equal_degree={self.equal_degree}, max_degree={self.max_degree})"
        )

    # ── construction ──────────────────────────────────────────────────
    @classmethod
    def from_parents(
        cls,
        parents,
        *,
        names: Sequence[str] | None = None,
    ) -> Topology:
        """Build a Topology from a BFS-ordered parents array.

        Parameters
        ----------
        parents :
            1-D array of int32; ``parents[i]`` is the index of node ``i``'s
            parent. ``parents[0]`` must be ``0`` (root self-parent).
            Must be BFS-ordered: ``parents[i] < i`` for ``i > 0``.
        names :
            Optional per-node names (informational only; used by Newick I/O).
        """
        return _build_topology(parents, names)


# ── builder ───────────────────────────────────────────────────────────
def _build_topology(parents_in, names) -> Topology:
    parents = np.asarray(parents_in, dtype=np.int32)
    if parents.ndim != 1:
        raise ValueError(f"parents must be 1-D, got shape {parents.shape}")
    if parents.size == 0:
        raise ValueError("parents must be non-empty")
    if int(parents[0]) != 0:
        raise ValueError("Convention: parents[0] must be 0 (root is its own parent).")
    n = int(parents.shape[0])

    # ── child counts: subtract the root's self-reference ──
    child_counts = np.bincount(parents, minlength=n).astype(np.int32)
    child_counts[0] -= 1

    # ── masks ──
    is_leaf = child_counts == 0
    is_root = np.zeros(n, dtype=bool)
    is_root[0] = True
    is_inner = ~is_leaf & ~is_root

    # ── node depths (BFS ordering guarantees parents[i] < i for i > 0) ──
    node_depths = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        p = int(parents[i])
        if p >= i:
            raise ValueError(
                f"parents array is not BFS-ordered at index {i} (parents[{i}]={p} >= {i})"
            )
        node_depths[i] = node_depths[p] + 1
    depth = int(node_depths.max()) if n > 0 else 0

    if n > 1 and not np.all(node_depths[1:] >= node_depths[:-1]):
        raise ValueError("parents array is not in BFS order (node_depths not monotone)")

    # ── level_starts: contiguous index range per level ──
    level_starts = np.zeros(depth + 2, dtype=np.int32)
    for d in range(depth + 1):
        level_starts[d + 1] = level_starts[d] + int((node_depths == d).sum())

    # ── degree info ──
    non_leaf_counts = child_counts[~is_leaf]
    equal_degree = bool(
        non_leaf_counts.size > 0 and bool(np.all(non_leaf_counts == non_leaf_counts[0]))
    )
    max_degree = int(child_counts.max())

    # ── segment-reduction layout: pbuckets / pbuckets_ref ──
    # For each level (taken bottom-up by the up-sweep), pvals = parent ids of the
    # nodes in this level. ``unique`` produces (a) the destination parent ids and
    # (b) the local segment id of each node within this level's reduction.
    pbuckets = parents.copy()
    pbuckets_ref: list = []
    for d in range(depth + 1):
        lo, hi = int(level_starts[d]), int(level_starts[d + 1])
        pvals = pbuckets[lo:hi]
        uniq, inv = np.unique(pvals, return_inverse=True)
        pbuckets[lo:hi] = inv.astype(np.int32)
        pbuckets_ref.append(uniq.astype(np.int32))

    return Topology(
        parents=parents,
        size=n,
        depth=depth,
        level_starts=level_starts,
        node_depths=node_depths,
        child_counts=child_counts,
        is_root=is_root,
        is_leaf=is_leaf,
        is_inner=is_inner,
        max_degree=max_degree,
        equal_degree=equal_degree,
        pbuckets=pbuckets,
        pbuckets_ref=tuple(pbuckets_ref),
        names=tuple(names) if names is not None else None,
    )


# ── JAX pytree registration ───────────────────────────────────────────
# Topology has no dynamic children; the whole object rides as static
# aux_data through jit. Two Topologies built from the same parents array
# compare equal and hit the same compile cache.
def _topo_flatten(t: Topology):
    return (), t


def _topo_unflatten(aux, _children):
    return aux


jax.tree_util.register_pytree_node(Topology, _topo_flatten, _topo_unflatten)
