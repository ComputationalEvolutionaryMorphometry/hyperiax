"""Newick (de)serialization in core.builders — pure-Python, no ete3."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperiax import Topology, Tree, from_newick, to_newick


@pytest.fixture
def restore_x64_config():
    original = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", original)


# ── basic round-trips ───────────────────────────────────────────────
def test_simple_newick_round_trip_bit_for_bit():
    """write(read(s)) == s for a simple unnamed-internal Newick."""
    src = "(A:1,(B:1,C:1):1);"
    tree = from_newick(src)
    assert to_newick(tree) == src


def test_complex_newick_round_trip_with_internal_names():
    """A more involved tree with named internal nodes and the root name 'N'.
    The root name is preserved (it carries a label but no branch length)."""
    src = "(((A:1,B:1)H:0.5,(C:1,D:2)I:0.5)L:0.1,((E:1,F:1,G:1)J:0.5,K:0.5)M:0.1)N;"
    tree = from_newick(src)
    assert to_newick(tree) == src


def test_topology_equality_after_round_trip():
    """Even on a write -> read -> write chain, topology must compare equal."""
    src = "(((A:1,B:1):0.5,(C:1,D:2):0.5):0.1,((E:1,F:1,G:1):0.5,K:0.5):0.1)N;"
    t1 = from_newick(src)
    t2 = from_newick(to_newick(t1))
    assert t1.topology == t2.topology
    assert jnp.allclose(t1["edge_length"], t2["edge_length"], atol=1e-6)
    assert t1.topology.names == t2.topology.names


# ── parsing details ────────────────────────────────────────────────
def test_read_extracts_edge_lengths():
    tree = from_newick("(A:0.5,B:1.5);")
    # BFS: root, A, B → edge_lengths in same order, root edge=0
    np.testing.assert_allclose(tree["edge_length"], [0.0, 0.5, 1.5])


def test_read_edge_length_dtype_tracks_jax_float_config(restore_x64_config):
    jax.config.update("jax_enable_x64", True)
    tree = from_newick("(A:0.1,B:0.2);")
    assert tree["edge_length"].dtype == np.dtype(jnp.float64)

    jax.config.update("jax_enable_x64", False)
    tree = from_newick("(A:0.1,B:0.2);")
    assert tree["edge_length"].dtype == np.dtype(jnp.float32)


def test_read_extracts_node_names_into_topology():
    tree = from_newick("(A:1,B:1)R;")
    assert tree.topology.names == ("R", "A", "B")


def test_read_unnamed_internal_nodes_use_empty_string():
    tree = from_newick("(A:1,(B:1,C:1):1);")
    # Root and the inner node have no names → empty strings.
    assert tree.topology.names == ("", "A", "", "B", "C")


def test_read_produces_bfs_ordered_parents():
    """The parents array hyperiax expects is BFS-ordered: parents[i] < i."""
    tree = from_newick("(((A:1,B:1):1,C:1):1,D:1);")
    parents = np.asarray(tree.topology.parents)
    assert parents[0] == 0
    for i in range(1, len(parents)):
        assert parents[i] < i, f"non-BFS at index {i}"


def test_read_handles_whitespace_between_tokens():
    tree = from_newick("( A:1 , ( B:1 , C:1 ):1 );")
    assert tree.topology.names == ("", "A", "", "B", "C")
    np.testing.assert_allclose(tree["edge_length"], [0.0, 1.0, 1.0, 1.0, 1.0])


def test_read_rejects_string_without_terminator():
    with pytest.raises(ValueError, match="';'"):
        from_newick("(A:1,B:1)")


# ── extra schema fields ────────────────────────────────────────────
def test_read_accepts_extra_schema_fields():
    tree = from_newick("(A:1,B:1);", schema={"value": (2,), "noise": (2,)})
    assert "edge_length" in tree.schema
    assert "value" in tree.schema
    assert "noise" in tree.schema
    # Extra fields allocated as zeros
    assert jnp.all(tree["value"] == 0)
    assert jnp.all(tree["noise"] == 0)


def test_read_extra_schema_does_not_disturb_edge_length():
    tree = from_newick("(A:0.5,B:1.5);", schema={"value": (2,)})
    np.testing.assert_allclose(tree["edge_length"], [0.0, 0.5, 1.5])


# ── file path reading ──────────────────────────────────────────────
def test_read_from_file_path(tmp_path: Path):
    src = "(A:1,(B:1,C:1):1);"
    nwk_file = tmp_path / "tree.nwk"
    nwk_file.write_text(src)
    tree = from_newick(nwk_file)
    assert to_newick(tree) == src


def test_read_from_file_path_as_string(tmp_path: Path):
    src = "(A:1,B:1);"
    nwk_file = tmp_path / "tree.nwk"
    nwk_file.write_text(src)
    tree = from_newick(str(nwk_file))
    assert to_newick(tree) == src


# ── write guards ───────────────────────────────────────────────────
def test_write_rejects_tree_without_edge_length():
    topo = Topology.from_parents([0, 0, 0])
    tree = Tree.empty(topo, {"value": ()})
    with pytest.raises(ValueError, match="edge_length"):
        to_newick(tree)


# ── interop with sweeps ────────────────────────────────────────────
def test_newick_tree_can_run_a_sweep():
    """Sanity: a Newick-parsed tree feeds into an up-sweep normally."""
    import hyperiax as hx

    tree = from_newick("(A:1,B:1,C:1)R;", schema={"value": ()})
    tree = tree.at[tree.topology.is_leaf].set(value=jnp.array([10.0, 20.0, 30.0]))

    @hx.up(reads_children=("value",), writes=("value",))
    def sum_up(node, children, params):
        return {"value": children.value.sum(0)}

    out = sum_up(tree)
    # Root has 3 children with values 10/20/30 → sum = 60
    assert jnp.allclose(out["value"][0], 60.0)
