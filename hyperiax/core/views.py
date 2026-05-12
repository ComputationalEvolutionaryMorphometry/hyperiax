"""Per-call views handed to user sweep functions.

Views are transient: the dispatcher constructs them anew at each level
(holding sliced/gathered JAX arrays), passes them to the user function,
and discards them when the function returns. They are *not* pytrees and
*not* hashable — they live entirely inside a trace.

Attribute access (``node.value``) is preferred over item access
(``node['value']``), but both are supported.
"""

from __future__ import annotations

from typing import Mapping

import jax


class Node:
    """Sliced per-node fields for one level (or one selected subset).

    Each attribute is a JAX array of shape ``(scope_size, *trailing)`` where
    ``scope_size`` is the number of nodes in this dispatch scope (typically
    a level's non-leaves).
    """

    __slots__ = ("_fields",)

    def __init__(self, fields: Mapping[str, jax.Array]):
        self._fields = dict(fields)

    def __getattr__(self, name: str):
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(
                f"Field {name!r} is not available in this scope. "
                f"Add it to `reads=...` on the sweep, or to the tree's schema. "
                f"Available: {sorted(self._fields)}"
            ) from None

    def __getitem__(self, name: str):
        return self._fields[name]

    def __iter__(self):
        return iter(self._fields)

    def __contains__(self, name) -> bool:
        return name in self._fields

    def __repr__(self) -> str:
        return f"Node(fields={sorted(self._fields)})"


class Children:
    """Per-parent view of children-of-each-node data.

    **Equal-degree mode (Stage 2):** each attribute is a real JAX array of
    shape ``(scope_size, k, *trailing)`` where ``k`` is the (constant) number
    of children per parent. Free to slice, index, multiply — anything you
    would do with a regular array. ``children.value.mean(0)`` averages over
    the ``k`` children of each parent.

    *Unequal-degree mode* arrives in Stage 4 via a ``ChildrenAxis`` proxy
    with the same surface for ``.sum/.max/.min/.prod/.mean(axis=0)``.
    """

    __slots__ = ("_fields",)

    def __init__(self, fields: Mapping[str, jax.Array]):
        self._fields = dict(fields)

    def __getattr__(self, name: str):
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(
                f"Children field {name!r} is not available in this scope. "
                f"Add it to `reads_children=...` on the sweep. "
                f"Available: {sorted(self._fields)}"
            ) from None

    def __getitem__(self, name: str):
        return self._fields[name]

    def __iter__(self):
        return iter(self._fields)

    def __contains__(self, name) -> bool:
        return name in self._fields

    def __repr__(self) -> str:
        return f"Children(fields={sorted(self._fields)})"
