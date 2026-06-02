"""Field-level schema for hyperiax Trees.

A :class:`Schema` is a hashable, ordered collection of named
:class:`FieldSpec` entries. It is the static contract for which arrays live
on a Tree and what their per-node shape/dtype must be. The Schema sits in
the JAX pytree aux_data of :class:`hyperiax.Tree`, so changing it
invalidates the JIT cache by design — fields are meant to be declared
up front via :meth:`Tree.empty`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


def _default_float_dtype() -> np.dtype:
    """Return JAX's current default floating dtype."""
    return np.dtype(jnp.asarray(0.0).dtype)


@dataclass(frozen=True)
class FieldSpec:
    """Per-node trailing shape and dtype of a single Tree field."""

    shape: tuple[int, ...]
    dtype: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        dtype = _default_float_dtype() if self.dtype is None else np.dtype(self.dtype)
        object.__setattr__(self, "dtype", dtype)

    @staticmethod
    def _coerce(spec) -> FieldSpec:
        """Coerce a ``FieldSpec``, a shape tuple, or ``None`` into a ``FieldSpec``."""
        if isinstance(spec, FieldSpec):
            return spec
        if spec is None:
            return FieldSpec(shape=())
        if isinstance(spec, tuple):
            return FieldSpec(shape=spec)
        raise TypeError(f"Cannot interpret {spec!r} as a FieldSpec")


@dataclass(frozen=True, eq=False)
class Schema:
    """Ordered, hashable collection of named FieldSpecs.

    Fields are stored as a ``tuple[(name, spec), ...]`` sorted by name so
    that equal sets of fields always hash identically regardless of insert
    order.
    """

    fields: tuple[tuple[str, FieldSpec], ...]

    # ── construction ──────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: Mapping[str, tuple | FieldSpec | None]) -> Schema:
        items = tuple(sorted((name, FieldSpec._coerce(spec)) for name, spec in d.items()))
        return cls(fields=items)

    # ── mapping-like surface ─────────────────────────────────────────
    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.fields)

    def __contains__(self, name: object) -> bool:
        return any(n == name for n, _ in self.fields)

    def __getitem__(self, name: str) -> FieldSpec:
        for n, spec in self.fields:
            if n == name:
                return spec
        raise KeyError(name)

    def __len__(self) -> int:
        return len(self.fields)

    def __iter__(self) -> Iterator[tuple[str, FieldSpec]]:
        return iter(self.fields)

    # ── identity ──────────────────────────────────────────────────────
    def __eq__(self, other) -> bool:
        return isinstance(other, Schema) and self.fields == other.fields

    def __hash__(self) -> int:
        return hash(self.fields)

    # ── functional updates ───────────────────────────────────────────
    def with_added(self, **extra) -> Schema:
        """Return a new Schema with extra fields appended. Raises if a name collides."""
        merged = dict(self.fields)
        for name, spec in extra.items():
            if name in merged:
                raise ValueError(f"Field {name!r} already in schema")
            merged[name] = FieldSpec._coerce(spec)
        return Schema.from_dict(merged)

    def without(self, *names: str) -> Schema:
        """Return a new Schema with the named fields removed."""
        kept = tuple((n, s) for n, s in self.fields if n not in names)
        return Schema(fields=kept)
