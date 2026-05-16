"""Newick read/write via :mod:`ete3`.

``ete3`` is an optional dependency (extra ``[io]``). The import is lazy
so that the rest of hyperiax remains usable without it.

Layout decisions
----------------
- Edge lengths land on ``tree.data['edge_length']`` (shape ``()``,
  ``float32``). The root has whatever distance ete3 reports — usually 0.
- Node names ride on :attr:`Topology.names` (a ``tuple[str, ...]``),
  *not* in ``tree.data``. Names are static topology metadata, not array
  data, so they don't belong in the JAX pytree.
- Extra schema fields can be requested via the ``schema`` argument to
  :func:`read`; they are allocated as zeros and the user fills them in
  afterwards.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..core.schema import FieldSpec, Schema
from ..core.topology import Topology
from ..core.tree import Tree


# ── public API ─────────────────────────────────────────────────────
def read(
    source: str | Path,
    *,
    schema: Schema | Mapping[str, tuple | FieldSpec | None] | None = None,
    newick_format: int = 1,
) -> Tree:
    """Read a Newick tree (literal or file path) into a hyperiax :class:`Tree`.

    Args:
        source: a Newick literal (string ending in ``;``) or a path to a
            Newick file.
        schema: optional extra fields beyond ``edge_length``.
        newick_format: ete3 format code; defaults to ``1`` (flexible with
            internal node names).

    Returns:
        A Tree whose schema always includes ``edge_length`` (``()``,
        float32) plus any extras requested.
    """
    ete3 = _require_ete3()
    # ete3.Tree accepts both Newick literals and file paths through the same ctor.
    ete_tree = ete3.Tree(str(source), format=newick_format)
    parents, names, edge_lengths = _ete_to_bfs_arrays(ete_tree)
    topo = Topology.from_parents(parents, names=names)

    merged: dict = {"edge_length": ()}
    if schema is not None:
        if isinstance(schema, Schema):
            for n, s in schema.fields:
                merged[n] = s
        else:
            merged.update(schema)
    full_schema = Schema.from_dict(merged)

    return Tree.empty(topo, full_schema).set(edge_length=jnp.asarray(edge_lengths))


def write(tree: Tree, *, newick_format: int = 1) -> str:
    """Convert a Tree back to a Newick string.

    Requires the Tree to have an ``edge_length`` field. Uses
    ``Topology.names`` for node labels (empty string for unnamed nodes).

    Args:
        tree: the Tree to serialize.
        newick_format: ete3 format code; defaults to ``1`` and should
            match what was used in :func:`read` for clean round-trips.
    """
    ete3 = _require_ete3()

    if "edge_length" not in tree.schema:
        raise ValueError(
            "Tree must have an 'edge_length' field to be written to Newick. "
            "Add it via Tree.update(edge_length=jnp.ones(tree.size)) first."
        )

    topo = tree.topology
    edge_lengths = np.asarray(tree["edge_length"])
    names: tuple[str, ...] = topo.names if topo.names is not None else ("",) * topo.size

    ete_nodes: list = [None] * topo.size
    root = ete3.Tree(name=names[0] or "", dist=float(edge_lengths[0]))
    ete_nodes[0] = root

    # BFS layout guarantees parents[i] < i, so parents are always built before children.
    for i in range(1, topo.size):
        parent_idx = int(topo.parents[i])
        child = ete_nodes[parent_idx].add_child(
            name=names[i] or "",
            dist=float(edge_lengths[i]),
        )
        ete_nodes[i] = child

    serialized = root.write(format=newick_format)
    # ete3 deliberately drops the root name in every format. Re-attach it
    # ourselves so that a name on the root survives a write→read round trip.
    if names[0]:
        # ete3 emits `...);` — insert root name just before the trailing `;`.
        assert serialized.endswith(";"), f"unexpected ete3 output: {serialized!r}"
        serialized = serialized[:-1] + names[0] + ";"
    return serialized


# ── helpers ────────────────────────────────────────────────────────
def _require_ete3():
    try:
        import ete3
    except ImportError as e:
        raise ImportError(
            "Newick I/O requires ete3. Install via `uv sync --extra io` "
            "or `pip install 'hyperiax[io]'`."
        ) from e
    return ete3


def _ete_to_bfs_arrays(
    ete_tree,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """BFS-traverse an ete3 Tree, returning ``(parents, names, edge_lengths)``.

    Root receives ``parents[0] == 0`` per the hyperiax convention. Edge
    length on the root is ``ete_tree.dist`` (usually 0).
    """
    parents: list[int] = []
    names: list[str] = []
    edge_lengths: list[float] = []

    # (node, parent_id_in_output | None for root)
    queue: deque = deque([(ete_tree, None)])
    while queue:
        node, parent_id = queue.popleft()
        my_id = len(parents)
        parents.append(my_id if parent_id is None else parent_id)
        names.append(node.name or "")
        edge_lengths.append(float(node.dist))
        for child in node.children:
            queue.append((child, my_id))

    return (
        np.asarray(parents, dtype=np.int32),
        tuple(names),
        np.asarray(edge_lengths, dtype=np.float32),
    )
