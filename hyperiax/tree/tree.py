from __future__ import annotations
from typing import Any, Tuple, Iterator, List, Optional
from collections import deque
from dataclasses import dataclass, field

from jax import numpy as jnp
import jax
from jax.typing import DTypeLike


@dataclass
class TopologyNode:
    parent: TopologyNode = None
    children: List[TopologyNode] = field(default_factory=list)
    _backref: HypTree = None
    name: str = None
    id: int = 0

    def __getitem__(self, arg: str) -> Any:
        if not self._backref:
            raise ValueError("This node is not part of a tree")
        return self._backref.data[arg][self.id]


class HypTree:
    def __init__(
        self,
        topology: TopologyNode,
        precompute_child_gathers: Optional[bool] = False,
        dtype: Optional[DTypeLike] = jnp.float32,
    ) -> None:
        self.topology_root = topology
        self.dtype = dtype

        # Indexing nodes in BFS order
        num_nodes = 0
        parents = [0]
        for i, node in enumerate(self.iter_topology_bfs()):
            num_nodes += 1
            node.id = i
            node._backref = self
            if node.parent:
                parents.append(node.parent.id)
            # child_counts.append(len(node.children))

        self.size = num_nodes
        self.data = {}  # container for all properties
        self.masks = {}  # masks for properties

        # Set up level structures
        self.levels = list(self._calculate_levels())
        self.node_depths = jnp.concatenate(
            [
                i * jnp.ones((lend - lstart, 1), dtype=int)
                for i, (lstart, lend) in enumerate(self.levels)
            ]
        )

        # Compute parent array, child counts, and node types
        self.parents = jnp.array(parents)
        self.child_counts = jax.ops.segment_sum(
            jnp.ones((self.size,), dtype=int),
            self.parents,
            num_segments=self.size,
            indices_are_sorted=True,
        )
        self.child_counts = self.child_counts.at[0].add(-1)
        self.is_leaf = ~self.child_counts.astype(bool)
        self.is_root = jnp.zeros((self.size,), dtype=bool)
        self.is_root = self.is_root.at[0].set(True)
        self.is_inner = ~self.is_leaf & ~self.is_root

        # Create level-local parent indices and references to global parent IDs
        pbuckets = jnp.copy(self.parents)
        pbuckets_ref = []
        for lmin, lmax in self.levels:
            pvals = pbuckets[lmin:lmax]
            pbuckets_refi, pbucket = jnp.unique(pvals, return_inverse=True)
            pbuckets = pbuckets.at[lmin:lmax].set(pbucket)
            pbuckets_ref.append(pbuckets_refi)
        self.pbuckets = pbuckets
        self.pbuckets_ref = pbuckets_ref

        # Tree metadata
        # self.child_counts = jnp.array(child_counts)
        self.depth = len(self.levels) - 1
        self.leaf_limits = self.levels[-1]

        self.coloring = jnp.zeros_like(self.parents, dtype=bool)
        for lmin, lmax in self.levels[::2]:
            self.coloring = self.coloring.at[lmin:lmax].set(True)

        # Precompute gather indices for children of each node if requested, only regular
        # trees supported (i.e. all nodes have the same number of children)
        if precompute_child_gathers:
            uniq = jnp.unique(self.child_counts)
            # len(uniq) == 2 implies that the tree has a "nice" structure (i.e. all nodes have the same number of children except for leaves, which have 0 children)
            if len(uniq) == 2:
                null, base = uniq
                assert null == 0, (
                    f"Only regular trees with the same number of children are supported, but got nodes with {null} children, which is not 0 or {base} for regular trees."
                )
                self.base = base
                # self.gather_child_idx = jnp.zeros((self.size, self.base), dtype=int)
                children = []
                for i, node in enumerate(self.iter_topology_bfs()):
                    if not node.children:
                        children.append(int(base) * [0])
                    else:
                        children.append([c.id for c in node.children])
                self.gather_child_idx = jnp.array(children)

                # Precompute non-leaf indices for each level
                self.level_non_leaf_indices = []
                for level_start, level_end in self.levels:
                    level_leaf_mask = self.is_leaf[level_start:level_end]
                    non_leaf_indices = jnp.where(~level_leaf_mask)[0] + level_start
                    self.level_non_leaf_indices.append(non_leaf_indices)
            else:
                raise ValueError(
                    f"Only trees with the same number of children are supported, but got nodes with the following number of children: {uniq}"
                )

    def _calculate_levels(self) -> Iterator[Tuple[int, int]]:
        """
        Calculate which nodes belong to each level (depth) of the tree and returns the start/end indices for each level.

        Yields:
            Iterator[Tuple[int, int]]: A tuple containing the start and end indices of each level.
        """
        queue = deque()
        buffer_queue = deque([self.topology_root])
        while queue or buffer_queue:
            if not queue:  # if queue is empty, flush the buffer and yield a level
                queue = buffer_queue
                yield (
                    buffer_queue[0].id,
                    buffer_queue[-1].id + 1,
                )  # to not pass the reference
                buffer_queue = deque()

            if children := queue.popleft().children:
                buffer_queue.extend(children)

    def __len__(self) -> int:
        """
        Return the number of nodes in the tree

        Returns:
            int: Number of nodes in the tree
        """
        return self.size

    def add_property(
        self,
        name: str,
        shape: Optional[Tuple[int, ...]] = (),
        dtype: Optional[jnp.dtype] = None,
    ):
        """
        Add a property with a given shape to the tree nodes. The added property
        plays as an initialization without any specific values. To set values, use
        in-place updates like `tree.data['property'] = ...`

        Args:
            name (str): Name of the property
            shape (Optional[Tuple[int, ...]], optional): Shape of the property. Defaults to ().
            dtype (Optional[jnp.dtype], optional): Data type of the property. Defaults to jnp.float32.
        """
        # TODO: Add custom initializers
        dtype = dtype or self.dtype
        self.data[name] = jnp.empty((self.size, *shape), dtype=dtype)
        self.masks[name] = jnp.zeros((self.size,), dtype=bool)

    def __repr__(self) -> str:
        """
        Return a string representation of the tree

        Returns:
            str: String representation of the tree
        """
        from hyperiax.plotting.ascii import TreeFormatter

        formatter = TreeFormatter(self)
        return f"HypTree:\nTopology:\n{formatter.format()}\nStatistics:\ntotal number of nodes = {self.size}\nnumber of levels = {len(self.levels)}\nnumber of leaves = {jnp.sum(self.is_leaf)}\nnumber of inner nodes = {jnp.sum(self.is_inner)}"

    def __str__(self) -> str:
        """
        Return a string representation of the tree

        Returns:
            str: String representation of the tree
        """
        return self.__repr__()

    ################################
    # Traversal methods for the tree
    ################################
    # dfs
    def iter_topology_dfs(self) -> Iterator[TopologyNode]:
        """
        Iterate over all of the nodes in the tree, in a depth-first manner.

        Yields:
            Iterator[TopologyNode]: The next node in the depth-first traversal.
        """
        stack = deque([self.topology_root])

        while stack:
            current = stack.pop()
            if current.children:
                stack.extend(reversed(current.children))
            yield current

    def iter_topology_leaves_dfs(self) -> Iterator[TopologyNode]:
        """
        Iterate over all of the leaves in the tree, in a depth-first manner.

        Yields:
            Iterator[TopologyNode]: The next leaf node in the depth-first traversal.
        """
        stack = deque([self.topology_root])

        while stack:
            current = stack.pop()
            if current.children:
                stack.extend(reversed(current.children))
            else:
                yield current

    # bfs
    def iter_topology_bfs(self) -> Iterator[TopologyNode]:
        """
        Iterate over all of the nodes in the tree, in a breadth-first manner.

        Yields:
            Iterator[TopologyNode]: The next node in the breadth-first traversal.
        """
        queue = deque([self.topology_root])

        while queue:
            current = queue.popleft()
            if current.children:
                queue.extend(current.children)
            yield current

    def iter_topology_leaves_bfs(self) -> Iterator[TopologyNode]:
        """
        Iterate over all of the leaves in the tree, in a breadth-first manner.

        Yields:
            Iterator[TopologyNode]: The next leaf node in the breadth-first traversal.
        """
        queue = deque([self.topology_root])

        while queue:
            current = queue.popleft()
            if current.children:
                queue.extend(current.children)
            else:
                yield current

    def iter_topology_levels(self) -> Iterator[List[TopologyNode]]:
        """
        Iterate over all of the nodes in the tree, level by level.

        Yields:
            Iterator[List[TopologyNode]]: The next level of nodes in the tree.
        """
        queue = deque()
        buffer_queue = deque([self.topology_root])
        while queue or buffer_queue:
            if not queue:  # if queue is empty, flush the buffer and yield a level
                queue = buffer_queue
                yield list(buffer_queue)  # to not pass the reference
                buffer_queue = deque()

            if children := queue.popleft().children:
                buffer_queue.extend(children)

    def iter_topology_post(self) -> Iterator[TopologyNode]:
        """
        Iterate over all of the nodes in the tree, in a post-order manner. Matching the Newick order.

        Yields:
            Iterator[TopologyNode]: The next node in the post-order traversal.
        """

        def post_order(node):
            for child in node.children:
                yield from post_order(child)
            yield node

        yield from post_order(self.topology_root)


class FastBiTree(HypTree):
    """
    Mostly for simulation purposes, creates a bifurcating tree
    without the topology, making instantiation fast
    """

    def __init__(self, depth) -> None:
        self.size = (2 << depth) - 1
        self.data = {}
        self.masks = {}

        self.levels = [(0, 1)] + [
            ((2 << i) - 1, (2 << (i + 1)) - 1) for i in range(depth)
        ]
        self.parents = jnp.concatenate([jnp.array([0]), jnp.arange(self.size - 1) // 2])
        self.depth = len(self.levels) - 1
        self.leaves = self.levels[-1]


# TODO: would love to get this work but https://github.com/google/jax/issues/4269 makes me believe this is probably a non JAX approach
class TreeField:
    def __init__(self, size, shape) -> None:
        raise NotImplementedError()
        self.shape = shape
        self.mask = jnp.zeros((size,), dtype=bool)
        self.data = jnp.empty((size, *shape))

    def __repr__(self) -> str:
        return str(self.data)
