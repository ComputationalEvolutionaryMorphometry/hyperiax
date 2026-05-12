"""Hyperiax core: pure-JAX tree primitives.

Strictly L1: only ``jax``, ``numpy``, and the standard library may be
imported from this package. Everything else (Newick I/O, prebuilt
models, plotting, MCMC) lives outside ``hyperiax.core``.
"""

from .builders import from_parents, symmetric_topology
from .errors import (
    HyperiaxError,
    MissingField,
    SchemaMismatch,
    StructureMismatch,
)
from .schema import FieldSpec, Schema
from .sweep import SweepFn, down, up
from .topology import Topology
from .tree import Tree
from .views import Children, ChildrenAxis, Node, Parent

__all__ = [
    "Children",
    "ChildrenAxis",
    "FieldSpec",
    "HyperiaxError",
    "MissingField",
    "Node",
    "Parent",
    "Schema",
    "SchemaMismatch",
    "StructureMismatch",
    "SweepFn",
    "Topology",
    "Tree",
    "down",
    "from_parents",
    "symmetric_topology",
    "up",
]
