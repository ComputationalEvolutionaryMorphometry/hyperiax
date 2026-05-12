"""Tree construction, mutation, and pytree behavior. Covers T-2, T-3."""

import jax
import jax.numpy as jnp
import pytest

from hyperiax import MissingField, SchemaMismatch, Topology, Tree


@pytest.fixture
def topo():
    return Topology.from_parents([0, 0, 0, 1, 1, 2, 2])


# ── construction ──────────────────────────────────────────────────────
def test_empty_allocates_zeros(topo):
    tree = Tree.empty(topo, {"value": (2,)})
    assert tree["value"].shape == (7, 2)
    assert jnp.all(tree["value"] == 0)


def test_from_data_infers_schema(topo):
    tree = Tree.from_data(topo, {"value": jnp.ones((7, 3))})
    assert tree.schema["value"].shape == (3,)
    assert jnp.all(tree["value"] == 1)


def test_from_data_rejects_wrong_leading_axis(topo):
    with pytest.raises(SchemaMismatch):
        Tree.from_data(topo, {"value": jnp.ones((5, 3))})


# ── set / set_at / shape validation (T-3) ───────────────────────────
def test_set_validates_shape(topo):
    tree = Tree.empty(topo, {"value": (2,)})
    with pytest.raises(SchemaMismatch):
        tree.set(value=jnp.zeros((7, 3)))  # wrong trailing shape


def test_set_rejects_wrong_size(topo):
    tree = Tree.empty(topo, {"value": (2,)})
    with pytest.raises(SchemaMismatch):
        tree.set(value=jnp.zeros((5, 2)))  # wrong leading axis


def test_set_rejects_unknown_field(topo):
    tree = Tree.empty(topo, {"value": (2,)})
    with pytest.raises(MissingField):
        tree.set(other=jnp.zeros((7, 2)))


def test_set_returns_new_tree_original_untouched(topo):
    tree = Tree.empty(topo, {"value": (2,)})
    new = tree.set(value=jnp.ones((7, 2)))
    assert tree is not new
    assert jnp.all(tree["value"] == 0)
    assert jnp.all(new["value"] == 1)


def test_set_at_with_boolean_mask(topo):
    tree = Tree.empty(topo, {"value": (2,)})
    leaf_vals = jnp.ones((int(topo.is_leaf.sum()), 2))
    new = tree.set_at(topo.is_leaf, value=leaf_vals)
    assert jnp.all(new["value"][topo.is_leaf] == 1)
    assert jnp.all(new["value"][~topo.is_leaf] == 0)


# ── update / drop ───────────────────────────────────────────────────
def test_update_extends_schema_with_spec(topo):
    tree = Tree.empty(topo, {"value": (2,)})
    extended = tree.update(noise=(2,))
    assert "noise" in extended.schema
    assert extended["noise"].shape == (7, 2)
    assert "noise" not in tree.schema  # original immutable


def test_update_extends_schema_with_array(topo):
    tree = Tree.empty(topo, {"value": (2,)})
    extended = tree.update(edges=jnp.arange(7, dtype=jnp.float32))
    assert extended.schema["edges"].shape == ()
    assert jnp.array_equal(extended["edges"], jnp.arange(7, dtype=jnp.float32))


def test_update_rejects_existing_field(topo):
    tree = Tree.empty(topo, {"value": (2,)})
    with pytest.raises(ValueError):
        tree.update(value=(3,))


def test_drop_removes_field(topo):
    tree = Tree.empty(topo, {"value": (2,), "noise": (2,)})
    smaller = tree.drop("noise")
    assert "noise" not in smaller.schema
    assert "value" in smaller.schema


# ── pytree round-trip (T-2) ──────────────────────────────────────────
def test_pytree_flatten_round_trips(topo):
    tree = (
        Tree.empty(topo, {"value": (2,), "edge": ()})
        .set(value=jnp.ones((7, 2)), edge=jnp.arange(7, dtype=jnp.float32))
    )
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    assert len(leaves) == 2
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, Tree)
    assert jnp.array_equal(rebuilt["value"], tree["value"])
    assert jnp.array_equal(rebuilt["edge"], tree["edge"])


def test_pytree_leaf_order_is_alphabetical_by_key(topo):
    tree = (
        Tree.empty(topo, {"value": (2,), "edge": (), "a_thing": (1,)})
        .set(
            value=jnp.full((7, 2), 2.0),
            edge=jnp.full((7,), 3.0),
            a_thing=jnp.full((7, 1), 1.0),
        )
    )
    leaves, _ = jax.tree_util.tree_flatten(tree)
    assert jnp.array_equal(leaves[0], jnp.full((7, 1), 1.0))
    assert jnp.array_equal(leaves[1], jnp.full((7,), 3.0))
    assert jnp.array_equal(leaves[2], jnp.full((7, 2), 2.0))


def test_pytree_structure_stable_across_topologically_identical_trees(topo):
    """Building 'the same' Tree twice must produce identical pytree structures —
    this is what lets JAX hit the JIT cache."""
    t1 = Tree.empty(topo, {"value": (2,)})
    t2 = Tree.empty(Topology.from_parents([0, 0, 0, 1, 1, 2, 2]), {"value": (2,)})
    _, d1 = jax.tree_util.tree_flatten(t1)
    _, d2 = jax.tree_util.tree_flatten(t2)
    assert d1 == d2


# ── identity ────────────────────────────────────────────────────────
def test_tree_is_not_hashable(topo):
    """Tree must not be hashable so users can't accidentally pass it as
    ``static_argnames`` — it must ride through JIT as a pytree."""
    tree = Tree.empty(topo, {"value": (2,)})
    with pytest.raises(TypeError):
        hash(tree)


def test_tree_equality_compares_data(topo):
    a = Tree.empty(topo, {"value": (2,)}).set(value=jnp.ones((7, 2)))
    b = Tree.empty(topo, {"value": (2,)}).set(value=jnp.ones((7, 2)))
    c = Tree.empty(topo, {"value": (2,)}).set(value=jnp.zeros((7, 2)))
    assert a == b
    assert a != c
