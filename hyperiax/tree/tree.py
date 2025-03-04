from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from jax import numpy as jnp
from typing import Iterator, List
from functools import partial
import time
import jax

@dataclass
class TopologyNode:
    parent : TopologyNode = None
    children : List[TopologyNode] = field(default_factory=list)
    _backref : HypTree = None
    name : str = None
    def __getitem__(self, arg):
        if not self._backref:
            raise ValueError("This node is not part of a tree")
        return self._backref.data[arg][self.id]
    
class HypTree:
    def __init__(self, topology, precompute_child_gathers = False) -> None:
        self.topology_root = topology

        nodes = 0
        parents = [0]
        for i,node in enumerate(self.iter_topology_bfs()):
            nodes += 1
            node.id = i
            node._backref = self
            if node.parent: 
                parents.append(node.parent.id)
            #child_counts.append(len(node.children)) 

        
        
        
        self.size = nodes
        self.data = {}
        self.masks = {}


        self.levels = list(self._calculate_levels())
        self.node_depths = jnp.concatenate([i*jnp.ones((lend-lstart, 1), dtype=int) for i,(lstart,lend) in enumerate(self.levels)])

        self.parents = jnp.array(parents)
        self.child_counts = jax.ops.segment_sum(jnp.ones((self.size,), dtype=int), self.parents, num_segments=self.size, indices_are_sorted=True)
        self.child_counts = self.child_counts.at[0].add(-1)
        self.is_leaf = ~self.child_counts.astype(bool)
        self.is_root = jnp.zeros((self.size,), dtype=bool)
        self.is_root = self.is_root.at[0].set(True)
        self.is_inner = ~self.is_leaf & ~self.is_root

        pbuckets = jnp.copy(self.parents)
        pbuckets_ref = []
        for lmin,lmax in self.levels:
            pvals = pbuckets[lmin:lmax]
            pbuckets_refi, pbucket = jnp.unique(pvals, return_inverse=True)
            pbuckets = pbuckets.at[lmin:lmax].set(pbucket)
            pbuckets_ref.append(pbuckets_refi)
        self.pbuckets = pbuckets
        self.pbuckets_ref = pbuckets_ref

        #self.child_counts = jnp.array(child_counts)
        self.depth = len(self.levels)-1
        self.leaf_limits = self.levels[-1]

        self.coloring = jnp.zeros_like(self.parents, dtype=bool)
        for lmin, lmax in self.levels[::2]:
            self.coloring = self.coloring.at[lmin:lmax].set(True)

        if precompute_child_gathers:
            uniq = jnp.unique(self.child_counts)
            if len(uniq) == 2:
                null, base = uniq
                assert null == 0
                self.base = base
                #self.gather_child_idx = jnp.zeros((self.size, self.base), dtype=int)
                children = []
                for i,node in enumerate(self.iter_topology_bfs()):
                    if not node.children: children.append(int(base)*[0])
                    else: children.append([c.id for c in node.children])
                self.gather_child_idx = jnp.array(children) 
            else:
                raise ValueError("Only trees with the same number of children are supported")
            # 2 unique values implies that the tree has a "nice" structure



    def _calculate_levels(self):
        queue = deque()
        buffer_queue = deque([self.topology_root])
        while queue or buffer_queue:
            if not queue: # if queue is empty, flush the buffer and yield a level
                queue = buffer_queue
                yield (buffer_queue[0].id, buffer_queue[-1].id+1) # to not pass the reference
                buffer_queue = deque()

            if children := queue.popleft().children:
                buffer_queue.extend(children)


    def __len__(self) -> int:
        return self.size

    def add_property(self, name, shape = (), dtype = None, initializer = None, key = None):
        self.data[name] = jnp.empty((self.size, *shape), dtype=dtype)
        self.masks[name] = jnp.zeros((self.size,), dtype=bool)

    def __repr__(self):
        """
        Return a string representation of the tree
        """
        return f"HypTree(size={self.size}, levels={len(self.levels)}, leaves={jnp.sum(self.is_leaf)}, inner nodes={jnp.sum(self.is_inner)})"

    def __str__(self):
        return self.__repr__()


    ################################
    # travelsal methods for the tree
    ################################
    # dfs
    def iter_topology_dfs(self) -> Iterator[TopologyNode]:
        """
        Iterate over all of the nodes in a depth-first manner.

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
        Iterate over all of the nodes in a breadth first manner

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
        Iterate over each level in the tree

        """
        queue = deque()
        buffer_queue = deque([self.topology_root])
        while queue or buffer_queue:
            if not queue: # if queue is empty, flush the buffer and yield a level
                queue = buffer_queue
                yield list(buffer_queue) # to not pass the reference
                buffer_queue = deque()

            if children := queue.popleft().children:
                buffer_queue.extend(children)   

    def iter_topology_post(self) -> Iterator[TopologyNode]:
        """
        Iterate over all nodes in post-order traversal (children before parents),
        matching the Newick format traversal order.
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
        self.size = (2<<depth)-1
        self.data = {}
        self.masks = {}

        self.levels = [(0,1)]+[((2<<i)-1, (2<<(i+1))-1) for i in range(depth)]
        self.parents = jnp.concatenate([jnp.array([0]),jnp.arange(self.size-1)//2])
        self.depth = len(self.levels)-1
        self.leaves = self.levels[-1]

## would love to get this work but https://github.com/google/jax/issues/4269 makes 
## me believe this is probably a non JAX approach
class TreeField:
    def __init__(self, size, shape) -> None:
        raise NotImplementedError()
        self.shape = shape
        self.mask = jnp.zeros((size,), dtype=bool)
        self.data = jnp.empty((size, *shape))
    def __repr__(self) -> str:
        return str(self.data)
