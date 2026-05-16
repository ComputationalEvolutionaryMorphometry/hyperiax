"""Tree — Topology + Schema + a dict of per-node arrays.

Immutable, JAX-pytree-registered: every mutator returns a new Tree, so
the Tree composes cleanly with ``@jax.jit`` and ``jax.lax.scan``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .errors import MissingField, SchemaMismatch
from .schema import FieldSpec, Schema
from .topology import Topology

# Field names that would shadow ``Tree`` attributes / methods under
# attribute access (``tree.value``). Schema field names are checked against
# this set at construction time so the shadowing fails loudly instead of
# silently masking the data.
_RESERVED_FIELD_NAMES = frozenset(
    {
        "topology",
        "schema",
        "data",
        "size",  # dataclass fields + property
        "set",
        "at",
        "update",
        "drop",  # mutators
        "empty",
        "from_data",  # classmethods
    }
)


@dataclass(frozen=True, eq=False)
class Tree:
    """A topology + a typed dict of ``(N, *trailing)`` arrays.

    Construction is always functional: every "mutator" returns a new Tree
    sharing the same Topology and (usually) Schema, with replaced data.

    Data fields are reachable two ways: ``tree['value']`` (dict-style, the
    safe form when the field name is dynamic or collides with a reserved
    name) and ``tree.value`` (attribute-style, the canonical form mirroring
    ``node.value`` / ``children.value`` inside a sweep).
    """

    topology: Topology
    schema: Schema
    data: dict  # dict[str, jax.Array]; keys match schema.names exactly

    # ── constructors ──────────────────────────────────────────────────
    @classmethod
    def empty(
        cls,
        topology: Topology,
        schema: Schema | Mapping[str, tuple | FieldSpec | None],
    ) -> Tree:
        """Allocate a Tree of zeros with the declared schema."""
        if not isinstance(schema, Schema):
            schema = Schema.from_dict(schema)
        cls._check_reserved(schema)
        n = topology.size
        data = {name: jnp.zeros((n, *spec.shape), dtype=spec.dtype) for name, spec in schema.fields}
        return cls(topology=topology, schema=schema, data=data)

    @classmethod
    def from_data(
        cls,
        topology: Topology,
        data: Mapping[str, jax.Array],
    ) -> Tree:
        """Build a Tree from an existing data dict; schema is inferred from arrays."""
        schema_dict = {}
        for name, arr in data.items():
            if arr.shape[0] != topology.size:
                raise SchemaMismatch(
                    f"Field {name!r}: leading axis {arr.shape[0]} != topology size {topology.size}"
                )
            schema_dict[name] = FieldSpec(shape=tuple(arr.shape[1:]), dtype=arr.dtype)
        schema = Schema.from_dict(schema_dict)
        cls._check_reserved(schema)
        return cls(topology=topology, schema=schema, data=dict(data))

    @classmethod
    def _check_reserved(cls, schema: Schema) -> None:
        conflicts = sorted(set(schema.names) & _RESERVED_FIELD_NAMES)
        if conflicts:
            raise SchemaMismatch(
                f"Schema field name(s) {conflicts} would shadow Tree attributes "
                f"under attribute access. Reserved names: {sorted(_RESERVED_FIELD_NAMES)}. "
                f"Rename your fields, or access via tree['name'] only."
            )

    # ── access ────────────────────────────────────────────────────────
    def __getitem__(self, name: str) -> jax.Array:
        try:
            return self.data[name]
        except KeyError:
            raise MissingField(
                f"Field {name!r} is not in this tree. Known fields: {self.schema.names}"
            ) from None

    def __getattr__(self, name: str) -> jax.Array:
        # Underscore names bail early to avoid pickle/repr/pytree recursion.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self.data[name]
        except KeyError:
            raise AttributeError(
                f"Tree has no field {name!r}. Schema: {self.schema.names}. "
                f"(Did you forget to add it via Tree.empty(topo, schema=...) or tree.update(...)?)"
            ) from None

    @property
    def size(self) -> int:
        return self.topology.size

    def __len__(self) -> int:
        return self.topology.size

    def __repr__(self) -> str:
        fields = ", ".join(f"{n}: {tuple(s.shape)}" for n, s in self.schema.fields)
        return f"Tree(size={self.size}, fields={{{fields}}})"

    # ── functional mutators ──────────────────────────────────────────
    def set(self, **fields: jax.Array) -> Tree:
        """Replace whole fields. Each value must already match ``(N, *spec.shape)``."""
        new_data = dict(self.data)
        for name, value in fields.items():
            if name not in self.schema:
                raise MissingField(
                    f"Cannot .set({name}=...); field is not in the schema. "
                    f"Use .update({name}=...) to add a brand-new field."
                )
            spec = self.schema[name]
            arr = jnp.asarray(value, dtype=spec.dtype)
            expected = (self.size, *spec.shape)
            if arr.shape != expected:
                raise SchemaMismatch(
                    f"Field {name!r}: expected shape {expected}, got {tuple(arr.shape)}"
                )
            new_data[name] = arr
        return Tree(topology=self.topology, schema=self.schema, data=new_data)

    @property
    def at(self) -> _AtAccessor:
        """JAX-style scatter accessor: ``tree.at[indices].set(field=values)``.

        Mirrors :attr:`jax.numpy.ndarray.at`. The returned indexer supports
        ``set``, ``add``, ``multiply``, ``min``, ``max`` (all take fields as
        keyword arguments and return a new :class:`Tree`) and ``get``
        (returns a ``dict`` of every field evaluated at ``indices``).
        """
        return _AtAccessor(self)

    def update(self, **fields) -> Tree:
        """Add brand-new fields. Values can be a FieldSpec, a shape tuple, or an array.

        Note: this changes the pytree structure and will invalidate any JIT
        caches keyed on the previous structure. Prefer ``Tree.empty(topo, full_schema)``
        when you know all the fields up front (the typical case in a sweep
        pipeline).
        """
        merged = dict(self.schema.fields)
        new_data = dict(self.data)
        for name, spec_or_value in fields.items():
            if name in self.schema:
                raise ValueError(f"Field {name!r} already in schema; use .set() to overwrite")
            if spec_or_value is None or isinstance(spec_or_value, (FieldSpec, tuple)):
                spec = FieldSpec._coerce(spec_or_value)
                merged[name] = spec
                new_data[name] = jnp.zeros((self.size, *spec.shape), dtype=spec.dtype)
            else:
                arr = jnp.asarray(spec_or_value)
                if arr.shape[0] != self.size:
                    raise SchemaMismatch(
                        f"Field {name!r}: leading axis {arr.shape[0]} != tree size {self.size}"
                    )
                merged[name] = FieldSpec(shape=tuple(arr.shape[1:]), dtype=arr.dtype)
                new_data[name] = arr
        new_schema = Schema.from_dict(merged)
        self._check_reserved(new_schema)
        return Tree(topology=self.topology, schema=new_schema, data=new_data)

    def drop(self, *names: str) -> Tree:
        """Drop fields; returns a Tree with reduced schema and data."""
        new_data = {k: v for k, v in self.data.items() if k not in names}
        return Tree(topology=self.topology, schema=self.schema.without(*names), data=new_data)

    # ── identity ──────────────────────────────────────────────────────
    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, Tree):
            return False
        if self.topology != other.topology or self.schema != other.schema:
            return False
        if set(self.data) != set(other.data):
            return False
        return all(jnp.array_equal(self.data[k], other.data[k]) for k in self.data)

    def __hash__(self):  # type: ignore[override]
        # Tree is never hashable: its dynamic data dict has no stable identity.
        # If you find yourself wanting to hash a Tree, you are probably trying
        # to pass it as a ``static_argnames`` — let it ride as a pytree instead.
        raise TypeError("Tree is not hashable; pass it as a JAX pytree, not as a static arg.")


# ── .at[...] accessor (JAX-style scatter for the multi-field Tree) ───
class _TreeIndexer:
    """Bound result of ``tree.at[idx]``; carries the indexer and the tree."""

    __slots__ = ("_tree", "_idx")

    def __init__(self, tree: Tree, idx) -> None:
        self._tree = tree
        self._idx = idx

    def _apply(self, op: str, fields: Mapping[str, jax.Array]) -> Tree:
        tree = self._tree
        new_data = dict(tree.data)
        for name, value in fields.items():
            if name not in tree.schema:
                raise MissingField(f"Cannot {op} unknown field {name!r}")
            spec = tree.schema[name]
            arr = jnp.asarray(value, dtype=spec.dtype)
            new_data[name] = getattr(new_data[name].at[self._idx], op)(arr)
        return Tree(topology=tree.topology, schema=tree.schema, data=new_data)

    def set(self, **fields: jax.Array) -> Tree:
        return self._apply("set", fields)

    def add(self, **fields: jax.Array) -> Tree:
        return self._apply("add", fields)

    def multiply(self, **fields: jax.Array) -> Tree:
        return self._apply("multiply", fields)

    def min(self, **fields: jax.Array) -> Tree:
        return self._apply("min", fields)

    def max(self, **fields: jax.Array) -> Tree:
        return self._apply("max", fields)

    def get(self) -> dict:
        """Return ``{field: tree.data[field][indices]}`` for every field."""
        return {n: self._tree.data[n][self._idx] for n in self._tree.schema.names}


class _AtAccessor:
    """Descriptor returned by :attr:`Tree.at`; indexes into a :class:`_TreeIndexer`."""

    __slots__ = ("_tree",)

    def __init__(self, tree: Tree) -> None:
        self._tree = tree

    def __getitem__(self, idx) -> _TreeIndexer:
        return _TreeIndexer(self._tree, idx)


# ── JAX pytree registration ───────────────────────────────────────────
# Leaves = data values sorted by key (deterministic for cache stability).
# aux_data = (topology, schema, key tuple).
def _tree_flatten(t: Tree):
    keys = tuple(sorted(t.data))
    leaves = tuple(t.data[k] for k in keys)
    aux = (t.topology, t.schema, keys)
    return leaves, aux


def _tree_unflatten(aux, leaves):
    topology, schema, keys = aux
    return Tree(topology=topology, schema=schema, data=dict(zip(keys, leaves)))


jax.tree_util.register_pytree_node(Tree, _tree_flatten, _tree_unflatten)
