from .executor import OrderedExecutor
from heapdict import heapdict


class DependencyTreeExecutor(OrderedExecutor):
    """Ordered executor that attempts to resolve dependencies in the tree.

    When batching it can include nodes from multiple levels in the same batch, 
    if they are not directly dependent on eachother with respect to the operation.
    """
    def _determine_execution_order(self, tree):
        heap = heapdict()

        mapping = dict()
        dependencies = dict()

        for i, node in enumerate(tree.iter_bfs()):
            node.id = i
            heap[i] = (len(node.children) if node.children else 0) + (1-(i+1)/len(tree)) #last term makes it vastly more efficient
            mapping[i] = node
            if node.parent:
                dependencies[i] = node.parent.id

        batches = []
        while len(heap) > 0:
            batch = []
            deps = []
            for i in range(self.batch_size):
                if len(heap) == 0: 
                    break
                k,p = heap.peekitem()
                if p < 1:
                    k,p = heap.popitem()
                    batch.append(mapping[k])
                    if k in dependencies.keys():
                        deps.append(dependencies[k])
                else:
                    break
            batches.append(batch)
            for d in deps:
                heap[d] -=1

        return list(reversed(batches))