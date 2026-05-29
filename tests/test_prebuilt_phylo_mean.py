"""phylo_mean prebuilt. Covers T-13 (cross-check against numpy reference)."""

import jax
import jax.numpy as jnp
import numpy as np

from hyperiax import Topology, Tree, symmetric_topology
from hyperiax.prebuilt import phylo_mean


# ── basic correctness (T-13) ────────────────────────────────────────
def test_phylo_mean_matches_hand_computation_on_7_node_tree():
    """7-node binary tree with non-uniform edges; the inner-node weighted
    means are hand-verifiable."""
    topo = Topology.from_parents([0, 0, 0, 1, 1, 2, 2])
    tree = Tree.empty(topo, {"estimated_value": (), "edge_length": ()})
    tree = tree.set(edge_length=jnp.array([0.0, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0]))
    tree = tree.at[topo.is_leaf].set(estimated_value=jnp.array([10.0, 20.0, 30.0, 40.0]))

    out = phylo_mean()(tree)

    # Hand-computed weighted means (see module-level docstring example)
    assert jnp.allclose(out["estimated_value"][1], 15.0)
    assert jnp.allclose(out["estimated_value"][2], 35.0)
    assert jnp.allclose(out["estimated_value"][0], 25.0)


def test_phylo_mean_leaves_untouched():
    topo = Topology.from_parents([0, 0, 0, 1, 1, 2, 2])
    leaves = jnp.array([10.0, 20.0, 30.0, 40.0])
    tree = (
        Tree.empty(topo, {"estimated_value": (), "edge_length": ()})
        .set(edge_length=jnp.ones(7))
        .at[topo.is_leaf]
        .set(estimated_value=leaves)
    )

    out = phylo_mean()(tree)
    assert jnp.array_equal(out["estimated_value"][topo.is_leaf], leaves)


def test_phylo_mean_uniform_edges_recovers_simple_mean():
    """When every edge has the same length, the weighted mean reduces to
    a simple recursive average — root equals the leaf mean."""
    topo = symmetric_topology(depth=3, degree=2)
    leaves = jnp.arange(8, dtype=jnp.float32)
    tree = (
        Tree.empty(topo, {"estimated_value": (), "edge_length": ()})
        .set(edge_length=jnp.ones(topo.size))
        .at[topo.is_leaf]
        .set(estimated_value=leaves)
    )

    out = phylo_mean()(tree)
    assert jnp.allclose(out["estimated_value"][0], leaves.mean())


def test_phylo_mean_with_vector_values():
    """Multi-dimensional ``estimated_value``: broadcasting must handle the
    trailing dims."""
    topo = symmetric_topology(depth=2, degree=2)
    leaves = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
    tree = (
        Tree.empty(topo, {"estimated_value": (2,), "edge_length": ()})
        .set(edge_length=jnp.ones(topo.size))
        .at[topo.is_leaf]
        .set(estimated_value=leaves)
    )

    out = phylo_mean()(tree)
    assert jnp.allclose(out["estimated_value"][0], leaves.mean(0))


# ── numpy reference cross-check (T-13 core) ─────────────────────────
def _numpy_phylo_mean(parents, edge_lengths, leaf_mask, leaf_values, depth, level_starts):
    """Reference implementation in plain numpy. Walks levels bottom-up,
    computing the edge-length-weighted mean at each parent."""
    n = parents.shape[0]
    vals = np.zeros(n, dtype=np.float32)
    vals[leaf_mask] = leaf_values
    for level in range(depth - 1, -1, -1):
        ls = int(level_starts[level])
        le = int(level_starts[level + 1])
        for parent_id in range(ls, le):
            child_ids = np.where(parents == parent_id)[0]
            child_ids = child_ids[child_ids != parent_id]  # drop self-ref on root
            if child_ids.size == 0:
                continue
            child_vals = vals[child_ids]
            child_edges = edge_lengths[child_ids]
            vals[parent_id] = (child_vals / child_edges).sum() / (1.0 / child_edges).sum()
    return vals


def test_phylo_mean_matches_numpy_reference_random():
    """Random edges and leaf values on a depth-3 binary tree must agree
    with a plain-numpy reference to floating-point tolerance."""
    topo = symmetric_topology(depth=3, degree=2)
    key = jax.random.PRNGKey(7)
    k_e, k_v = jax.random.split(key)
    edge_lengths = jax.random.uniform(k_e, (topo.size,), minval=0.1, maxval=2.0)
    leaf_vals = jax.random.normal(k_v, (int(topo.is_leaf.sum()),))

    tree = (
        Tree.empty(topo, {"estimated_value": (), "edge_length": ()})
        .set(edge_length=edge_lengths)
        .at[topo.is_leaf]
        .set(estimated_value=leaf_vals)
    )
    hx_result = np.asarray(phylo_mean()(tree)["estimated_value"])

    np_result = _numpy_phylo_mean(
        parents=np.asarray(topo.parents),
        edge_lengths=np.asarray(edge_lengths),
        leaf_mask=np.asarray(topo.is_leaf),
        leaf_values=np.asarray(leaf_vals),
        depth=topo.depth,
        level_starts=np.asarray(topo.level_starts),
    )
    np.testing.assert_allclose(hx_result, np_result, atol=1e-5, rtol=1e-5)


# ── pipeline composition ───────────────────────────────────────────
def test_phylo_mean_composes_under_outer_jit():
    topo = symmetric_topology(depth=2, degree=2)
    tree = (
        Tree.empty(topo, {"estimated_value": (), "edge_length": ()})
        .set(edge_length=jnp.ones(topo.size))
        .at[topo.is_leaf]
        .set(estimated_value=jnp.array([1.0, 2.0, 3.0, 4.0]))
    )
    sweep = phylo_mean()

    eager = sweep(tree)
    jitted = jax.jit(sweep)(tree)
    assert jnp.allclose(eager["estimated_value"], jitted["estimated_value"])


def test_phylo_mean_runs_on_newick_tree():
    """End-to-end: read a tree from Newick, run phylo_mean."""
    from hyperiax import from_newick

    src = "((A:1,B:1):2,(C:0.5,D:1.5):0.5);"
    tree = from_newick(src, schema={"estimated_value": ()})
    # Leaf indices in BFS order: identify leaves and seed.
    leaf_vals = jnp.array([1.0, 2.0, 3.0, 4.0])
    tree = tree.at[tree.topology.is_leaf].set(estimated_value=leaf_vals)

    out = phylo_mean()(tree)
    # Sanity: root estimate is some finite weighted mean of leaves.
    root_est = float(out["estimated_value"][0])
    assert min(leaf_vals.tolist()) <= root_est <= max(leaf_vals.tolist())


# ── unequal-degree works via children.map ──────────────────────────
def test_phylo_mean_on_unequal_degree_tree():
    """Ragged-arity trees are supported: per-child work goes through
    children.map, the fusion through segment reductions."""
    # node 1 has 3 children (3,4,5); node 2 has 2 children (6,7); root has 2 (1,2).
    topo = Topology.from_parents([0, 0, 0, 1, 1, 1, 2, 2])
    assert not topo.equal_degree
    leaves = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])  # nodes 3..7
    tree = (
        Tree.empty(topo, {"estimated_value": (), "edge_length": ()})
        .set(edge_length=jnp.array([0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 2.0, 2.0]))
        .at[topo.is_leaf]
        .set(estimated_value=leaves)
    )

    out = phylo_mean()(tree)

    # Hand reference per parent:  sum_c (v_c / l_c) / sum_c (1 / l_c).
    def wmean(vs, ls):
        return float((vs / ls).sum() / (1.0 / ls).sum())

    n1 = wmean(leaves[:3], jnp.array([1.0, 1.0, 1.0]))  # children 3,4,5
    n2 = wmean(leaves[3:], jnp.array([2.0, 2.0]))       # children 6,7
    root = wmean(jnp.array([n1, n2]), jnp.array([0.5, 0.5]))
    assert jnp.allclose(out["estimated_value"][1], n1)
    assert jnp.allclose(out["estimated_value"][2], n2)
    assert jnp.allclose(out["estimated_value"][0], root)
    # Leaves untouched.
    assert jnp.array_equal(out["estimated_value"][topo.is_leaf], leaves)
