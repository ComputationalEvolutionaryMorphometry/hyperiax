"""Internal dispatch engine.

The dispatcher walks a Tree's levels in the right order, builds per-level
:class:`Node`/:class:`Children` views over sliced/gathered JAX arrays,
calls the user's sweep function (under :func:`jax.vmap` for the down sweep
and the equal-degree up sweep), and scatters the writes into a new data
dict. The whole thing is a pure ``Tree -> Tree`` function so it composes
with ``jax.jit`` and ``jax.lax.scan`` without leaking state.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from .errors import MissingField, SchemaMismatch
from .tree import Tree
from .views import Children, ChildrenAxis, Node, Parent

if TYPE_CHECKING:
    from .sweep import SweepFn


# ── public entrypoints ──────────────────────────────────────────────
def up_dispatch(sweep: SweepFn, tree: Tree, params, key) -> Tree:
    """Run an up-sweep on a Tree. Returns a new Tree.

    Picks one of two internal paths based on the topology:

    - **Equal-degree** (every non-leaf has the same arity): use
      ``topo.gather_child_idx`` to build a dense ``(scope, k, *trailing)``
      array per field, then :func:`jax.vmap` the user function over the
      scope axis so it sees one parent at a time.
    - **Unequal-degree**: use the level's ``pbuckets`` / ``pbuckets_ref``
      to feed each ``children.X`` access through a :class:`ChildrenAxis`
      proxy that dispatches reductions to ``jax.ops.segment_*``.
    """
    _validate_schema(sweep, tree)
    if tree.topology.equal_degree:
        return _up_dispatch_equal(sweep, tree, params, key)
    return _up_dispatch_unequal(sweep, tree, params, key)


def down_dispatch(sweep: SweepFn, tree: Tree, params, key) -> Tree:
    """Run a down-sweep on a Tree. Returns a new Tree.

    Works for arbitrary topologies — every non-root node has exactly one
    parent, so the dispatcher slices each level contiguously and gathers
    the parent record via ``topo.parents``.
    """
    _validate_schema(sweep, tree)
    return _down_dispatch_jit(sweep, tree, params, key)


# ── shared helpers ─────────────────────────────────────────────────
def _resolve_reads(
    sweep_reads: tuple | None,
    sweep_other_reads: tuple | None,
    schema_names: tuple,
) -> tuple[tuple, tuple, frozenset]:
    """Resolve ``reads_self``, ``reads_other``, and ``expected_writes`` for a sweep."""
    reads_self = sweep_reads if sweep_reads is not None else schema_names
    reads_other = sweep_other_reads if sweep_other_reads is not None else schema_names
    return reads_self, reads_other


def _check_writes(out, expected: frozenset, declared: tuple, *, direction: str) -> None:
    got = frozenset(out)
    if got != expected:
        extra = sorted(got - expected)
        missing = sorted(expected - got)
        raise SchemaMismatch(
            f"{direction} returned keys {sorted(got)} but writes={sorted(declared)}. "
            f"Extra: {extra}; missing: {missing}."
        )


# ── equal-degree up dispatch ────────────────────────────────────────
@partial(jax.jit, static_argnums=(0,))
def _up_dispatch_equal(sweep: SweepFn, tree: Tree, params, key) -> Tree:
    """Up sweep on a tree where every non-leaf has the same arity.

    Each level builds a dense ``(scope, k, *trailing)`` children array via
    ``gather_child_idx``; :func:`jax.vmap` lifts the user fn to a
    per-parent view (``node.value : (*trailing,)``, ``children.value :
    (k, *trailing)``).
    """
    topo = tree.topology
    data = dict(tree.data)
    reads_self, reads_children = _resolve_reads(
        sweep.reads,
        sweep.reads_children,
        tree.schema.names,
    )
    expected_writes = frozenset(sweep.writes)

    per_level = jax.vmap(
        lambda nd, cd, p: sweep.fn(Node(nd), Children(cd), p),
        in_axes=(0, 0, None),
    )

    for level in range(topo.depth - 1, -1, -1):
        non_leaf = topo.level_non_leaf_indices[level]
        if non_leaf.size == 0:
            continue

        self_idx = jnp.asarray(non_leaf)
        child_idx = jnp.asarray(topo.gather_child_idx[non_leaf])

        node_data = {k: data[k][self_idx] for k in reads_self}
        children_data = {k: data[k][child_idx] for k in reads_children}

        out = per_level(node_data, children_data, params)
        _check_writes(out, expected_writes, sweep.writes, direction="Up-sweep")

        for k, v in out.items():
            data[k] = data[k].at[self_idx].set(v)

    return Tree(topology=topo, schema=tree.schema, data=data)


# ── unequal-degree up dispatch (segment-reduction path) ────────────
@partial(jax.jit, static_argnums=(0,))
def _up_dispatch_unequal(sweep: SweepFn, tree: Tree, params, key) -> Tree:
    """Up sweep on a tree with ragged arity.

    At each parent level, the children one level deeper are concatenated
    into a flat ``(M_total, *trailing)`` view and assigned segment ids via
    ``topo.pbuckets``. The user's ``children.X.sum(0)`` (or .max/.mean/...)
    dispatches to the corresponding ``jax.ops.segment_*`` through a
    :class:`ChildrenAxis` proxy. No vmap is needed — proxy reductions
    return per-parent shape ``(num_parents, *trailing)`` directly.
    """
    topo = tree.topology
    data = dict(tree.data)
    reads_self, reads_children = _resolve_reads(
        sweep.reads,
        sweep.reads_children,
        tree.schema.names,
    )
    expected_writes = frozenset(sweep.writes)
    schema = tree.schema

    for level in range(topo.depth - 1, -1, -1):
        child_ls = int(topo.level_starts[level + 1])
        child_le = int(topo.level_starts[level + 2])
        parent_ids = topo.pbuckets_ref[level + 1]
        num_segments = int(parent_ids.size)
        if num_segments == 0:
            continue

        parent_idx_jax = jnp.asarray(parent_ids)
        seg_ids = jnp.asarray(topo.pbuckets[child_ls:child_le])

        node_data = {k: data[k][parent_idx_jax] for k in reads_self}
        children_data = {
            k: ChildrenAxis(
                flat=data[k][child_ls:child_le],
                segments=seg_ids,
                num_segments=num_segments,
                trailing=schema[k].shape,
            )
            for k in reads_children
        }

        out = sweep.fn(Node(node_data), Children(children_data), params)
        _check_writes(out, expected_writes, sweep.writes, direction="Up-sweep")

        for k, v in out.items():
            data[k] = data[k].at[parent_idx_jax].set(v)

    return Tree(topology=topo, schema=tree.schema, data=data)


# ── down dispatch ──────────────────────────────────────────────────
@partial(jax.jit, static_argnums=(0,))
def _down_dispatch_jit(sweep: SweepFn, tree: Tree, params, key) -> Tree:
    topo = tree.topology
    data = dict(tree.data)
    reads_self, reads_parent = _resolve_reads(
        sweep.reads,
        sweep.reads_parent,
        tree.schema.names,
    )
    expected_writes = frozenset(sweep.writes)

    per_level = jax.vmap(
        lambda nd, pd, p: sweep.fn(Node(nd), Parent(pd), p),
        in_axes=(0, 0, None),
    )

    # Walk root → leaves; level 1 is first (the root has no parent).
    for level in range(1, topo.depth + 1):
        ls = int(topo.level_starts[level])
        le = int(topo.level_starts[level + 1])

        parent_indices = jnp.asarray(topo.parents[ls:le])

        node_data = {k: data[k][ls:le] for k in reads_self}
        parent_data = {k: data[k][parent_indices] for k in reads_parent}

        out = per_level(node_data, parent_data, params)
        _check_writes(out, expected_writes, sweep.writes, direction="Down-sweep")

        for k, v in out.items():
            data[k] = data[k].at[ls:le].set(v)

    return Tree(topology=topo, schema=tree.schema, data=data)


# ── validation (runs eagerly, outside the jit-cached body) ─────────
def _validate_schema(sweep: SweepFn, tree: Tree) -> None:
    """Check that every field referenced by the sweep exists in the schema.

    Runs eagerly so the error message has a clean Python traceback to the
    decorator site — much friendlier than a JAX KeyError several frames deep.
    """
    needed: set[str] = set()
    if sweep.reads is not None:
        needed.update(sweep.reads)
    if sweep.reads_children is not None:
        needed.update(sweep.reads_children)
    if sweep.reads_parent is not None:
        needed.update(sweep.reads_parent)
    needed.update(sweep.writes)

    missing = needed - set(tree.schema.names)
    if missing:
        raise MissingField(
            f"Sweep references field(s) {sorted(missing)} that are not in the tree's schema. "
            f"Schema has: {tree.schema.names}. "
            f"Add them via tree.update(...) or include them in Tree.empty(topo, schema)."
        )
