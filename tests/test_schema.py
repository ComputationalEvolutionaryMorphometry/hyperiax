"""Schema / FieldSpec sanity tests."""

import jax.numpy as jnp
import pytest

from hyperiax import FieldSpec, Schema


def test_fieldspec_coerce_from_tuple():
    spec = FieldSpec._coerce((2, 3))
    assert spec.shape == (2, 3)


def test_fieldspec_coerce_from_none_is_scalar():
    assert FieldSpec._coerce(None).shape == ()


def test_fieldspec_coerce_rejects_garbage():
    with pytest.raises(TypeError):
        FieldSpec._coerce(object())


def test_schema_from_dict_orders_fields_by_name():
    s = Schema.from_dict({"z": (2,), "a": (3,), "m": ()})
    assert s.names == ("a", "m", "z")


def test_schema_is_value_equal_and_hashable():
    a = Schema.from_dict({"x": (2,), "y": (3,)})
    b = Schema.from_dict({"y": (3,), "x": (2,)})  # different insert order
    assert a == b
    assert hash(a) == hash(b)


def test_schema_contains_and_getitem():
    s = Schema.from_dict({"x": (2,)})
    assert "x" in s
    assert "y" not in s
    assert s["x"].shape == (2,)
    with pytest.raises(KeyError):
        _ = s["y"]


def test_schema_with_added_returns_new_and_keeps_original():
    s = Schema.from_dict({"x": (2,)})
    s2 = s.with_added(y=(3,))
    assert s2.names == ("x", "y")
    assert s.names == ("x",)  # original immutable
    with pytest.raises(ValueError):
        s.with_added(x=(4,))


def test_schema_without_drops_fields():
    s = Schema.from_dict({"x": (2,), "y": (3,), "z": ()})
    assert s.without("y").names == ("x", "z")


def test_schema_empty():
    assert Schema.empty().names == ()
    assert len(Schema.empty()) == 0
