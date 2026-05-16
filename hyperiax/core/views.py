"""Per-call views handed to user sweep functions.

Views are transient: the dispatcher constructs them anew at each level
(holding sliced/gathered JAX arrays), passes them to the user function,
and discards them when the function returns. They are *not* pytrees and
*not* hashable — they live entirely inside a trace.

Attribute access (``node.value``) is preferred over item access
(``node['value']``), but both are supported.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from .errors import HyperiaxError


class _FieldsView:
    """Shared base for per-call field-dict views.

    Provides attribute / item / iteration / containment access over a fixed
    dict of arrays. Subclasses set ``_kind`` (used in error messages and
    repr) and a tag for which decorator argument adds missing fields.
    """

    __slots__ = ("_fields",)
    _kind: str = "Fields"
    _reads_arg: str = "reads"

    def __init__(self, fields: Mapping[str, jax.Array]):
        self._fields = dict(fields)

    def __getattr__(self, name: str):
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(
                f"{self._kind} field {name!r} is not available in this scope. "
                f"Add it to `{self._reads_arg}=...` on the sweep. "
                f"Available: {sorted(self._fields)}"
            ) from None

    def __getitem__(self, name: str):
        return self._fields[name]

    def __iter__(self):
        return iter(self._fields)

    def __contains__(self, name) -> bool:
        return name in self._fields

    def __repr__(self) -> str:
        return f"{self._kind}(fields={sorted(self._fields)})"


class Node(_FieldsView):
    """Sliced per-node fields for one level (or one selected subset).

    Each attribute is a JAX array of shape ``(scope_size, *trailing)`` where
    ``scope_size`` is the number of nodes in this dispatch scope (typically
    a level's non-leaves).
    """

    __slots__ = ()
    _kind = "Node"
    _reads_arg = "reads"


class Parent(_FieldsView):
    """Per-node view of the parent's fields. Under :func:`jax.vmap`, each
    attribute is a JAX array of shape ``(*trailing,)`` — one parent record
    per node at the current level."""

    __slots__ = ()
    _kind = "Parent"
    _reads_arg = "reads_parent"


class Children(_FieldsView):
    """Per-parent view of children-of-each-node data.

    **Equal-degree mode:** each attribute is a real JAX array of shape
    ``(scope_size, k, *trailing)`` where ``k`` is the (constant) number of
    children per parent. Free to slice, index, multiply — anything you would
    do with a regular array. ``children.value.mean(0)`` averages over the
    ``k`` children of each parent.

    **Unequal-degree mode:** each attribute is a :class:`ChildrenAxis` proxy
    exposing the same reduction surface (``.sum/.max/.min/.prod/.mean``) but
    dispatching to :func:`jax.ops.segment_*`. User code is identical to the
    equal-degree case.
    """

    __slots__ = ()
    _kind = "Children"
    _reads_arg = "reads_children"


class ChildrenAxis:
    """Virtual children-axis proxy for unequal-degree trees.

    Backs a flat ``(M_total, *trailing)`` JAX array (every child at the
    current level concatenated) plus a ``segments`` array assigning each
    row to a parent. ``num_segments`` is a *static* Python int (derived
    from the topology) — required for ``jax.ops.segment_*`` to produce a
    statically-shaped output.

    The user calls one of ``.sum / .prod / .max / .min / .mean (axis=0)``
    to reduce over the children axis. Each dispatches to the matching
    segment-reduction. Output shape is ``(num_segments, *trailing)``,
    matching the corresponding ``Node`` view at the same level.

    Non-reduction ops (indexing, broadcasted arithmetic, NumPy coercion)
    are deliberately rejected: they would either silently produce wrong
    results on the flat layout or imply a padded dense form the user
    should request explicitly via :meth:`Children.gather` (TODO).
    """

    __slots__ = ("_flat", "_segments", "_num_segments", "_trailing")

    def __init__(
        self,
        flat: jax.Array,
        segments: jax.Array,
        num_segments: int,
        trailing: tuple,
    ):
        self._flat = flat
        self._segments = segments
        self._num_segments = num_segments
        self._trailing = tuple(trailing)

    # ── reductions ────────────────────────────────────────────────────
    def sum(self, axis: int = 0) -> jax.Array:
        self._check_axis(axis)
        return jax.ops.segment_sum(
            self._flat,
            self._segments,
            num_segments=self._num_segments,
            indices_are_sorted=True,
        )

    def prod(self, axis: int = 0) -> jax.Array:
        self._check_axis(axis)
        return jax.ops.segment_prod(
            self._flat,
            self._segments,
            num_segments=self._num_segments,
            indices_are_sorted=True,
        )

    def max(self, axis: int = 0) -> jax.Array:
        self._check_axis(axis)
        return jax.ops.segment_max(
            self._flat,
            self._segments,
            num_segments=self._num_segments,
            indices_are_sorted=True,
        )

    def min(self, axis: int = 0) -> jax.Array:
        self._check_axis(axis)
        return jax.ops.segment_min(
            self._flat,
            self._segments,
            num_segments=self._num_segments,
            indices_are_sorted=True,
        )

    def mean(self, axis: int = 0) -> jax.Array:
        self._check_axis(axis)
        sums = self.sum(0)
        counts = jax.ops.segment_sum(
            jnp.ones_like(self._segments, dtype=self._flat.dtype),
            self._segments,
            num_segments=self._num_segments,
            indices_are_sorted=True,
        )
        # Reshape counts to broadcast against the trailing dims.
        counts = counts.reshape((self._num_segments,) + (1,) * len(self._trailing))
        return sums / counts

    # ── guards ────────────────────────────────────────────────────────
    @staticmethod
    def _check_axis(axis: int) -> None:
        if axis != 0:
            raise ValueError(
                f"ChildrenAxis reductions must use axis=0 (the children axis); got axis={axis}."
            )

    def __array__(self, dtype=None):
        raise HyperiaxError(
            "ChildrenAxis is a virtual proxy on an unequal-degree tree; it "
            "cannot be coerced to a dense array. Reduce first via "
            ".sum/.prod/.max/.min/.mean(axis=0), or fall back to "
            "children.gather() for a padded view."
        )

    def __repr__(self) -> str:
        return (
            f"ChildrenAxis(num_segments={self._num_segments}, "
            f"trailing={self._trailing}, flat_size={self._flat.shape[0]})"
        )
