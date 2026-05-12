"""Hyperiax core: pure-JAX tree primitives.

Strictly L1: only ``jax``, ``numpy``, and the standard library may be
imported from this package. Everything else (Newick I/O, prebuilt
models, plotting, MCMC) lives outside ``hyperiax.core``.
"""

from .errors import (
    HyperiaxError,
    MissingField,
    SchemaMismatch,
    StructureMismatch,
)
from .schema import FieldSpec, Schema
from .topology import Topology
from .tree import Tree

__all__ = [
    "FieldSpec",
    "HyperiaxError",
    "MissingField",
    "Schema",
    "SchemaMismatch",
    "StructureMismatch",
    "Topology",
    "Tree",
]
