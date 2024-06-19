from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from jax import numpy as jnp
from typing import Iterator, List
from functools import partial

@dataclass
class TopologyNode:
    parent : TopologyNode = None
    children : List[TopologyNode] = field(default_factory=list)
    _backref : FastTree = None

    def __getitem__(self, arg):
        if not self._backref:
            raise ValueError("This node is not part of a tree")
        return self._backref.data[arg][self.id]
    
class FastTree:
    def __init__(self, topology) -> None:
        self.topology_root = topology

        nodes = 0
        parents = [0]
        #child_counts = []
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
        #self.psizes = [end-start for start,end in self.levels]
        self.parents = jnp.array(parents)
        pbuckets = jnp.copy(self.parents)
        pbuckets_ref = []
        for lmin,lmax in self.levels:
            pvals = pbuckets[lmin:lmax]
            pbuckets_refi, pbucket = self._truncate_bucket(pvals)
            pbuckets = pbuckets.at[lmin:lmax].set(pbucket)
            pbuckets_ref.append(pbuckets_refi)
        self.pbuckets = pbuckets
        self.pbuckets_ref = pbuckets_ref

        #self.child_counts = jnp.array(child_counts)
        self.depth = len(self.levels)-1
        self.leaves = self.levels[-1]

    def _truncate_bucket(self, bucket):
        vals = [bucket[0]]
        outbuck = [0]
        cur = bucket[0]
        skipped = bucket[0]
        for el in bucket[1:]:
            if el > cur: 
                skipped += el-cur-1
                cur = el
                vals.append(el)
            outbuck.append(el-skipped)
        return jnp.array(vals), jnp.array(outbuck)

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

    def add_property(self, name, shape, initializer = None, key = None):
        self.data[name] = jnp.empty((self.size, *shape))
        self.masks[name] = jnp.zeros((self.size,), dtype=bool)

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


class FastBiTree(FastTree):
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