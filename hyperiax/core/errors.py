"""Hyperiax exception hierarchy."""


class HyperiaxError(Exception):
    """Base class for all hyperiax-raised exceptions."""


class SchemaMismatch(HyperiaxError):
    """A value's shape or dtype does not match the field's spec."""


class MissingField(HyperiaxError):
    """A sweep or tree access references a name not in the tree's schema."""
