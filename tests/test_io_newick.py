"""Newick I/O round-trips via ete3. Covers T-12."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

# Skip the entire module gracefully if ete3 isn't installed.
ete3 = pytest.importorskip("ete3")

from hyperiax import Topology, Tree
from hyperiax.io import newick


# ── basic round-trips ───────────────────────────────────────────────
def test_simple_newick_round_trip_bit_for_bit():
    """T-12: write(read(s)) == s for a simple unnamed-internal Newick."""
    src = "(A:1,(B:1,C:1):1);"
    tree = newick.read(src)
    assert newick.write(tree) == src


def test_complex_newick_round_trip_with_internal_names():
    """A more involved tree with named internal nodes and the root name 'N'.
    The root name is preserved (despite ete3 dropping it by default —
    our writer patches it back)."""
    src = "(((A:1,B:1)H:0.5,(C:1,D:2)I:0.5)L:0.1,((E:1,F:1,G:1)J:0.5,K:0.5)M:0.1)N;"
    tree = newick.read(src)
    assert newick.write(tree) == src


def test_topology_equality_after_round_trip():
    """Even on a write→read→write chain, topology must compare equal."""
    src = "(((A:1,B:1):0.5,(C:1,D:2):0.5):0.1,((E:1,F:1,G:1):0.5,K:0.5):0.1)N;"
    t1 = newick.read(src)
    t2 = newick.read(newick.write(t1))
    assert t1.topology == t2.topology
    assert jnp.allclose(t1["edge_length"], t2["edge_length"], atol=1e-6)
    assert t1.topology.names == t2.topology.names


# ── parsing details ────────────────────────────────────────────────
def test_read_extracts_edge_lengths():
    tree = newick.read("(A:0.5,B:1.5);")
    # BFS: root, A, B → edge_lengths in same order, root edge=0
    np.testing.assert_allclose(tree["edge_length"], [0.0, 0.5, 1.5])


def test_read_extracts_node_names_into_topology():
    tree = newick.read("(A:1,B:1)R;")
    assert tree.topology.names == ("R", "A", "B")


def test_read_unnamed_internal_nodes_use_empty_string():
    tree = newick.read("(A:1,(B:1,C:1):1);")
    # Root and the inner node have no names → empty strings.
    assert tree.topology.names == ("", "A", "", "B", "C")


def test_read_produces_bfs_ordered_parents():
    """The parents array hyperiax expects is BFS-ordered: parents[i] < i."""
    tree = newick.read("(((A:1,B:1):1,C:1):1,D:1);")
    parents = np.asarray(tree.topology.parents)
    assert parents[0] == 0
    for i in range(1, len(parents)):
        assert parents[i] < i, f"non-BFS at index {i}"


# ── extra schema fields ────────────────────────────────────────────
def test_read_accepts_extra_schema_fields():
    tree = newick.read("(A:1,B:1);", schema={"value": (2,), "noise": (2,)})
    assert "edge_length" in tree.schema
    assert "value" in tree.schema
    assert "noise" in tree.schema
    # Extra fields allocated as zeros
    assert jnp.all(tree["value"] == 0)
    assert jnp.all(tree["noise"] == 0)


def test_read_extra_schema_does_not_disturb_edge_length():
    tree = newick.read("(A:0.5,B:1.5);", schema={"value": (2,)})
    np.testing.assert_allclose(tree["edge_length"], [0.0, 0.5, 1.5])


# ── file path reading ──────────────────────────────────────────────
def test_read_from_file_path(tmp_path: Path):
    src = "(A:1,(B:1,C:1):1);"
    nwk_file = tmp_path / "tree.nwk"
    nwk_file.write_text(src)
    tree = newick.read(nwk_file)
    assert newick.write(tree) == src


# ── write guards ───────────────────────────────────────────────────
def test_write_rejects_tree_without_edge_length():
    topo = Topology.from_parents([0, 0, 0])
    tree = Tree.empty(topo, {"value": ()})
    with pytest.raises(ValueError, match="edge_length"):
        newick.write(tree)


# ── interop with sweeps ────────────────────────────────────────────
def test_newick_tree_can_run_a_sweep():
    """Sanity: a Newick-parsed tree feeds into an up-sweep normally."""
    import hyperiax as hx

    tree = newick.read("(A:1,B:1,C:1)R;", schema={"value": ()})
    tree = tree.at[tree.topology.is_leaf].set(value=jnp.array([10.0, 20.0, 30.0]))

    @hx.up(reads_children=("value",), writes=("value",))
    def sum_up(node, children, params):
        return {"value": children.value.sum(0)}

    out = sum_up(tree)
    # Root has 3 children with values 10/20/30 → sum = 60
    assert jnp.allclose(out["value"][0], 60.0)
