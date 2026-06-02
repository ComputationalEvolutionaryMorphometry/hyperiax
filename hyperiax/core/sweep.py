"""Sweep decorators — turn a user function into a Tree → Tree transform.

A :class:`SweepFn` carries a direction (``up`` / ``down``), the user
function, and the explicit ``reads`` / ``writes`` declarations. It is
``frozen`` and hashable, so the dispatcher can use it as a JIT
``static_argnums`` and reuse compilations across calls.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import jax

from .tree import Tree


def _normalize(seq) -> tuple[str, ...] | None:
    if seq is None:
        return None
    return tuple(seq)


@dataclass(frozen=True, eq=False)
class SweepFn:
    """A direction-tagged, pure ``Tree -> Tree`` transform.

    Hashable so the dispatcher can pin it as JIT static. Equality is
    value-based on ``(direction, fn identity, read/write spec tuples)``.

    ``writes_children`` (up sweeps only) declares fields whose per-parent
    output has shape ``(k, *trailing)`` and gets scattered to the
    children's slots rather than the parent's slot — useful for caching
    per-edge quantities computed during the up-sweep (e.g. the per-step
    ``F_t``/``H_t`` from a per-edge ODE filter).
    """

    direction: Literal["up", "down"]
    fn: Callable
    reads: tuple[str, ...] | None
    reads_children: tuple[str, ...] | None
    reads_parent: tuple[str, ...] | None
    writes: tuple[str, ...]
    writes_children: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.writes and not self.writes_children:
            raise ValueError(
                "@up / @down requires writes=(...) (or writes_children=(...) on @up) "
                "with at least one field"
            )
        if self.direction == "up" and self.reads_parent is not None:
            raise ValueError("up sweeps cannot reference parent fields; use @down for that.")
        if self.direction == "down" and self.reads_children is not None:
            raise ValueError("down sweeps cannot reference children fields; use @up for that.")
        if self.direction == "down" and self.writes_children:
            raise ValueError(
                "writes_children is only valid on up sweeps; down sweeps write per-node only."
            )
        overlap = set(self.writes) & set(self.writes_children)
        if overlap:
            raise ValueError(
                f"writes and writes_children share fields {sorted(overlap)}; "
                "each field must be one or the other."
            )

    # ── identity ──────────────────────────────────────────────────────
    def __hash__(self) -> int:
        # `id(self.fn)` is sufficient: two functions with different identity
        # mean different traced code, so a different jit cache entry is correct.
        return hash(
            (
                self.direction,
                id(self.fn),
                self.reads,
                self.reads_children,
                self.reads_parent,
                self.writes,
                self.writes_children,
            )
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SweepFn)
            and self.direction == other.direction
            and self.fn is other.fn
            and self.reads == other.reads
            and self.reads_children == other.reads_children
            and self.reads_parent == other.reads_parent
            and self.writes == other.writes
            and self.writes_children == other.writes_children
        )

    def __repr__(self) -> str:
        extra = f", writes_children={self.writes_children}" if self.writes_children else ""
        return (
            f"SweepFn({self.direction}, "
            f"fn={getattr(self.fn, '__name__', repr(self.fn))}, "
            f"writes={self.writes}{extra})"
        )

    # ── invocation ────────────────────────────────────────────────────
    def __call__(
        self,
        tree: Tree,
        *,
        params: Mapping | None = None,
        key: jax.Array | None = None,
    ) -> Tree:
        # Local import avoids a circular at module import time.
        from .dispatch import down_dispatch, up_dispatch

        if params is None:
            params = {}
        if self.direction == "up":
            return up_dispatch(self, tree, params, key)
        if self.direction == "down":
            return down_dispatch(self, tree, params, key)
        raise NotImplementedError(f"{self.direction!r} dispatch not yet wired up")


# ── decorators ──────────────────────────────────────────────────────
def up(
    *,
    reads: Sequence[str] | None = None,
    reads_children: Sequence[str] | None = None,
    writes: Sequence[str] = (),
    writes_children: Sequence[str] = (),
) -> Callable[[Callable], SweepFn]:
    """Decorator: mark a function as an up-sweep.

    The decorated function has signature ``(node, children, params)`` and
    must return a ``dict[str, Array]`` whose key set equals
    ``writes ∪ writes_children`` exactly. Values for keys in ``writes``
    have shape ``(*trailing,)`` per parent (under :func:`jax.vmap`);
    values for keys in ``writes_children`` have shape ``(k, *trailing)``
    per parent and are scattered to the children's slots.

    At least one of ``writes`` or ``writes_children`` must be non-empty.

    Args:
        reads: Fields the function reads from the current node. ``None``
            (the default) means "all schema fields at call time."
        reads_children: Fields the function reads from each child.
            ``None`` defaults to all schema fields at call time.
        writes: Fields written back per parent (canonical case).
        writes_children: Fields written per child (one row per outgoing
            edge). Use when an up sweep also produces per-edge byproducts
            you want to cache on each child (e.g. per-edge ODE-filter
            trajectories).

    Example::

        @hx.up(reads_children=('value',), writes=('value',))
        def avg(node, children, params):
            return {'value': children.value.mean(0)}

        new_tree = avg(tree)
    """

    def _wrap(fn: Callable) -> SweepFn:
        return SweepFn(
            direction="up",
            fn=fn,
            reads=_normalize(reads),
            reads_children=_normalize(reads_children),
            reads_parent=None,
            writes=tuple(writes),
            writes_children=tuple(writes_children),
        )

    return _wrap


def down(
    *,
    reads: Sequence[str] | None = None,
    reads_parent: Sequence[str] | None = None,
    writes: Sequence[str],
) -> Callable[[Callable], SweepFn]:
    """Decorator: mark a function as a down-sweep.

    The decorated function has signature ``(node, parent, params)`` and
    must return a ``dict[str, Array]`` whose key set equals ``writes``
    exactly. Each returned value must have shape ``(*trailing,)`` matching
    the schema for that field (per-node under :func:`jax.vmap`).

    The root is never visited — it has no parent. Seed its data with
    ``tree.set(...)`` before calling the sweep.

    Args:
        reads: Fields the function reads from the current node. ``None``
            defaults to all schema fields at call time.
        reads_parent: Fields the function reads from each node's parent.
            ``None`` defaults to all schema fields at call time.
        writes: Required. The fields the function writes back into the
            tree. Must be non-empty.

    Example::

        @hx.down(reads=('delta',), reads_parent=('value',), writes=('value',))
        def propagate(node, parent, params):
            return {'value': parent.value + node.delta}

        new_tree = propagate(tree)
    """

    def _wrap(fn: Callable) -> SweepFn:
        return SweepFn(
            direction="down",
            fn=fn,
            reads=_normalize(reads),
            reads_children=None,
            reads_parent=_normalize(reads_parent),
            writes=tuple(writes),
        )

    return _wrap
