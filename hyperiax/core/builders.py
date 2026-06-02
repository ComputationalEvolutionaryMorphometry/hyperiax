"""Topology and tree builders.

Constructors that produce a fully-derived :class:`Topology` (or a
:class:`Tree` carrying per-node data) from a compact description:

- :func:`symmetric_topology` / :func:`from_parents` build a bare topology.
- :func:`from_newick` / :func:`to_newick` (de)serialize Newick strings,
  carrying branch lengths on ``tree.data['edge_length']``.

The Newick reader/writer is a small pure-Python implementation — no
external parser — so it stays inside the L1 ``jax + numpy + stdlib``
boundary that ``hyperiax.core`` is allowed to import from.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .schema import FieldSpec, Schema
from .topology import Topology
from .tree import Tree


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
            n_at_level = degree**level
            local = np.arange(n_at_level, dtype=np.int32)
            parents[cursor : cursor + n_at_level] = prev_level_start + local // degree
            prev_level_start = cursor
            cursor += n_at_level

    return Topology.from_parents(parents)


def from_parents(parents, *, names=None) -> Topology:
    """Thin re-export of :meth:`Topology.from_parents`."""
    return Topology.from_parents(parents, names=names)


# ── Newick I/O ─────────────────────────────────────────────────────
def from_newick(
    source: str | Path,
    *,
    schema: Schema | Mapping[str, tuple | FieldSpec | None] | None = None,
) -> Tree:
    """Read a Newick tree (literal or file path) into a hyperiax :class:`Tree`.

    Branch lengths land on ``tree.data['edge_length']`` (shape ``()``) with
    the current default floating dtype unless ``schema`` overrides it; the
    root edge length is whatever the Newick string reports (usually 0). Node
    names ride on :attr:`Topology.names` (static metadata), not in
    ``tree.data``.

    Args:
        source: a Newick literal (string ending in ``;``) or a path to a
            Newick file.
        schema: optional extra fields beyond ``edge_length``; allocated as
            zeros for the caller to fill in afterwards.

    Returns:
        A Tree whose schema always includes ``edge_length`` (``()``) plus any
        extras requested.
    """
    root = _parse_newick(_read_newick_source(source))
    parents, names, edge_lengths = _bfs_arrays(root)
    topo = Topology.from_parents(parents, names=names)

    merged: dict = {"edge_length": ()}
    if schema is not None:
        if isinstance(schema, Schema):
            for name, spec in schema.fields:
                merged[name] = spec
        else:
            merged.update(schema)
    full_schema = Schema.from_dict(merged)

    return Tree.empty(topo, full_schema).set(edge_length=edge_lengths)


def to_newick(tree: Tree) -> str:
    """Serialize a Tree back to a Newick string.

    Requires an ``edge_length`` field. Uses :attr:`Topology.names` for node
    labels (empty string for unnamed nodes). The root carries a label but no
    branch length, so a name on the root survives a write -> read round trip.

    Args:
        tree: the Tree to serialize.
    """
    if "edge_length" not in tree.schema:
        raise ValueError(
            "Tree must have an 'edge_length' field to be written to Newick. "
            "Add it via Tree.update(edge_length=jnp.ones(tree.size)) first."
        )

    topo = tree.topology
    parents = np.asarray(topo.parents)
    edge_lengths = np.asarray(tree["edge_length"])
    names: tuple[str, ...] = topo.names if topo.names is not None else ("",) * topo.size

    # BFS layout guarantees parents[i] < i, so a single pass builds the
    # children adjacency in left-to-right insertion order.
    children: list[list[int]] = [[] for _ in range(topo.size)]
    for i in range(1, topo.size):
        children[int(parents[i])].append(i)

    def serialize(i: int, is_root: bool) -> str:
        if children[i]:
            inner = ",".join(serialize(c, False) for c in children[i])
            label = f"({inner}){names[i]}"
        else:
            label = names[i]
        if not is_root:
            label += f":{_format_length(edge_lengths[i])}"
        return label

    return serialize(0, True) + ";"


# ── Newick helpers ─────────────────────────────────────────────────
class _NewickNode:
    """Mutable parse-tree node: a name, a branch length, and children."""

    __slots__ = ("name", "length", "children")

    def __init__(self) -> None:
        self.name: str = ""
        self.length: float = 0.0
        self.children: list[_NewickNode] = []


def _read_newick_source(source: str | Path) -> str:
    """Return the Newick text for a literal or a file path."""
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8")
    text = str(source)
    # A literal ends in ';' or contains a '(' group; everything else (a bare
    # name with no markup) is treated as a file path. Parsing a malformed
    # literal then surfaces a clear error rather than a FileNotFoundError.
    stripped = text.strip()
    if stripped.endswith(";") or "(" in stripped:
        return text
    return Path(text).read_text(encoding="utf-8")


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t\n\r":
        i += 1
    return i


def _parse_newick(text: str) -> _NewickNode:
    """Parse a Newick string into a nested :class:`_NewickNode` tree.

    Supports the common grammar (matching ete3's flexible format): nested
    ``(...)`` groups, optional internal/root labels, and ``:length`` branch
    lengths. Quoted labels are not handled.
    """
    text = text.strip()
    if not text.endswith(";"):
        raise ValueError(f"Newick string must end with ';', got ...{text[-20:]!r}")

    node, pos = _parse_subtree(text, 0)
    pos = _skip_ws(text, pos)
    if pos >= len(text) or text[pos] != ";":
        raise ValueError(f"malformed Newick: expected ';' terminator at position {pos}")
    return node


def _parse_subtree(s: str, i: int) -> tuple[_NewickNode, int]:
    i = _skip_ws(s, i)
    node = _NewickNode()
    if i < len(s) and s[i] == "(":
        i += 1
        while True:
            child, i = _parse_subtree(s, i)
            node.children.append(child)
            i = _skip_ws(s, i)
            if i >= len(s):
                raise ValueError("malformed Newick: unterminated '(' group")
            if s[i] == ",":
                i += 1
                continue
            if s[i] == ")":
                i += 1
                break
            raise ValueError(f"malformed Newick: unexpected {s[i]!r} at position {i}")
    return _parse_label(s, i, node)


def _parse_label(s: str, i: int, node: _NewickNode) -> tuple[_NewickNode, int]:
    i = _skip_ws(s, i)
    start = i
    while i < len(s) and s[i] not in ":,();":
        i += 1
    node.name = s[start:i].strip()

    i = _skip_ws(s, i)
    if i < len(s) and s[i] == ":":
        i += 1
        i = _skip_ws(s, i)
        start = i
        while i < len(s) and s[i] not in ",();":
            i += 1
        node.length = float(s[start:i].strip())
    return node, i


def _bfs_arrays(root: _NewickNode) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """BFS-traverse a parse tree into ``(parents, names, edge_lengths)``.

    Root receives ``parents[0] == 0`` per the hyperiax convention; BFS order
    guarantees ``parents[i] < i`` for all other nodes.
    """
    parents: list[int] = []
    names: list[str] = []
    edge_lengths: list[float] = []

    queue: deque[tuple[_NewickNode, int | None]] = deque([(root, None)])
    while queue:
        node, parent_id = queue.popleft()
        my_id = len(parents)
        parents.append(my_id if parent_id is None else parent_id)
        names.append(node.name)
        edge_lengths.append(node.length)
        for child in node.children:
            queue.append((child, my_id))

    return (
        np.asarray(parents, dtype=np.int32),
        tuple(names),
        np.asarray(edge_lengths),
    )


def _format_length(x) -> str:
    """Format a branch length minimally (``1.0`` -> ``1``, ``0.5`` -> ``0.5``).

    Matches the 6-significant-figure ``%g`` style, which also keeps common
    round-off artifacts compact.
    """
    return format(float(x), "g")
