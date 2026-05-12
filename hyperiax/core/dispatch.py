"""Internal dispatch engine.

The dispatcher walks a Tree's levels in the right order, builds per-level
:class:`Node`/:class:`Children` views over sliced/gathered JAX arrays,
calls the user's sweep function under :func:`jax.vmap` (so the function
sees one parent at a time), and scatters the writes back into a new data
dict. The whole thing is a pure ``Tree -> Tree`` function so it composes
with ``jax.jit`` and ``jax.lax.scan`` without leaking state.

Stage 2 implements the equal-degree up-sweep only. Stage 3 adds down. Stage 4
adds the unequal-degree segment-reduction path with a ``ChildrenAxis`` proxy.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from .errors import MissingField, SchemaMismatch
from .tree import Tree
from .views import Children, Node, Parent

if TYPE_CHECKING:
    from .sweep import SweepFn


# ── public entrypoints ──────────────────────────────────────────────
def up_dispatch(sweep: "SweepFn", tree: Tree, params, key) -> Tree:
    """Run an up-sweep on a Tree. Returns a new Tree."""
    _validate_schema(sweep, tree)
    if not tree.topology.equal_degree:
        raise NotImplementedError(
            "Unequal-degree up-sweep is not implemented yet (arrives in Stage 4). "
            f"Topology has non-leaf child counts "
            f"{sorted(set(tree.topology.child_counts[~tree.topology.is_leaf].tolist()))}."
        )
    return _up_dispatch_jit(sweep, tree, params, key)


def down_dispatch(sweep: "SweepFn", tree: Tree, params, key) -> Tree:
    """Run a down-sweep on a Tree. Returns a new Tree.

    Works for arbitrary topologies — the down direction does not need the
    equal-degree fast path because every non-root node has exactly one
    parent. The dispatcher slices the level contiguously and gathers parent
    data via ``topo.parents``.
    """
    _validate_schema(sweep, tree)
    return _down_dispatch_jit(sweep, tree, params, key)


# ── jit-able inner body ─────────────────────────────────────────────
@partial(jax.jit, static_argnums=(0,))
def _up_dispatch_jit(sweep: "SweepFn", tree: Tree, params, key) -> Tree:
    topo = tree.topology
    data = dict(tree.data)

    schema_names = tuple(name for name, _ in tree.schema.fields)
    reads_self = sweep.reads if sweep.reads is not None else schema_names
    reads_children = sweep.reads_children if sweep.reads_children is not None else schema_names
    expected_writes = frozenset(sweep.writes)

    # User function is written from a *single parent's* point of view:
    #   node.value     : (*trailing,)
    #   children.value : (k, *trailing)
    # We vmap it over the level's batch dimension.
    def _per_parent(node_d, children_d, params_):
        return sweep.fn(Node(node_d), Children(children_d), params_)

    per_level = jax.vmap(_per_parent, in_axes=(0, 0, None))

    # Walk parent levels from the deepest non-leaf level down to the root.
    # depth-0 trees have no work to do (root only).
    for level in range(topo.depth - 1, -1, -1):
        non_leaf = topo.level_non_leaf_indices[level]
        if non_leaf.size == 0:
            continue

        self_idx = jnp.asarray(non_leaf)
        child_idx = jnp.asarray(topo.gather_child_idx[non_leaf])  # (scope, k)

        node_data = {k: data[k][self_idx] for k in reads_self}
        children_data = {k: data[k][child_idx] for k in reads_children}

        out = per_level(node_data, children_data, params)

        got = frozenset(out)
        if got != expected_writes:
            extra = sorted(got - expected_writes)
            missing = sorted(expected_writes - got)
            raise SchemaMismatch(
                f"Up-sweep returned keys {sorted(got)} but writes={sorted(sweep.writes)}. "
                f"Extra: {extra}; missing: {missing}."
            )

        for k, v in out.items():
            data[k] = data[k].at[self_idx].set(v)

    return Tree(topology=topo, schema=tree.schema, data=data)


@partial(jax.jit, static_argnums=(0,))
def _down_dispatch_jit(sweep: "SweepFn", tree: Tree, params, key) -> Tree:
    topo = tree.topology
    data = dict(tree.data)

    schema_names = tuple(name for name, _ in tree.schema.fields)
    reads_self = sweep.reads if sweep.reads is not None else schema_names
    reads_parent = sweep.reads_parent if sweep.reads_parent is not None else schema_names
    expected_writes = frozenset(sweep.writes)

    # User function is per-node under jax.vmap:
    #   node.value   : (*trailing,)
    #   parent.value : (*trailing,)
    def _per_node(node_d, parent_d, params_):
        return sweep.fn(Node(node_d), Parent(parent_d), params_)

    per_level = jax.vmap(_per_node, in_axes=(0, 0, None))

    # Walk levels root → leaves, starting at level 1 (root has no parent).
    for level in range(1, topo.depth + 1):
        ls = int(topo.level_starts[level])
        le = int(topo.level_starts[level + 1])

        parent_indices = jnp.asarray(topo.parents[ls:le])  # (scope,)

        node_data = {k: data[k][ls:le] for k in reads_self}
        parent_data = {k: data[k][parent_indices] for k in reads_parent}

        out = per_level(node_data, parent_data, params)

        got = frozenset(out)
        if got != expected_writes:
            extra = sorted(got - expected_writes)
            missing = sorted(expected_writes - got)
            raise SchemaMismatch(
                f"Down-sweep returned keys {sorted(got)} but writes={sorted(sweep.writes)}. "
                f"Extra: {extra}; missing: {missing}."
            )

        for k, v in out.items():
            data[k] = data[k].at[ls:le].set(v)

    return Tree(topology=topo, schema=tree.schema, data=data)


# ── validation (runs eagerly, outside the jit-cached body) ─────────
def _validate_schema(sweep: "SweepFn", tree: Tree) -> None:
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
